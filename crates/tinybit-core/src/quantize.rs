use candle_core::{Device, DType, Tensor};

/// Marker tensor name present in a tinybit quantized-export safetensors file.
/// `TinyBit::load` detects it and reconstructs full-precision weights.
pub const QUANT_MARKER: &str = "__tinybit_quant__";

/// Quantize a 2D weight matrix to packed ternary for export.
/// Returns (packed_bytes, scale, rows, cols). Two ternary values per byte, so
/// the packed payload is ~1/16 the size of the original f32 matrix.
pub fn quantize_pack_2d(w: &Tensor) -> anyhow::Result<(Vec<u8>, f32, usize, usize)> {
    let (rows, cols) = w.dims2()?;
    let (tern, scale) = quantize_ternary(w)?;
    let flat: Vec<f32> = tern.reshape((rows * cols,))?.to_vec1::<f32>()?;
    let i8v: Vec<i8> = flat.iter().map(|&v| v as i8).collect();
    Ok((pack_ternary(&i8v), scale, rows, cols))
}

/// Reconstruct a full-precision (f32) 2D weight from packed ternary + scale.
/// Inverse of [`quantize_pack_2d`]; the result is `scale * {-1,0,+1}`.
pub fn dequantize_unpack_2d(
    packed: &[u8],
    scale: f32,
    rows: usize,
    cols: usize,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let i8v = unpack_ternary(packed, rows * cols);
    let f: Vec<f32> = i8v.iter().map(|&v| v as f32 * scale).collect();
    Ok(Tensor::from_vec(f, (rows, cols), device)?)
}

/// Quantize a weight matrix to ternary {-1, 0, +1}.
/// Uses the mean absolute value as the threshold (BitNet b1.58 method).
/// Returns (quantized_weights: Tensor[i8], scale: f32).
pub fn quantize_ternary(w: &Tensor) -> anyhow::Result<(Tensor, f32)> {
    let w_f32 = w.to_dtype(DType::F32)?;
    let abs_w = w_f32.abs()?;
    let scale = abs_w.mean_all()?.to_scalar::<f32>()?;
    if scale == 0.0 {
        let zeros = Tensor::zeros_like(&w_f32)?;
        let qi8 = zeros.to_dtype(DType::I64)?.to_dtype(DType::F32)?;
        return Ok((qi8, scale));
    }
    // Threshold: values above mean-abs become +1, below -mean-abs become -1, else 0
    let threshold = Tensor::full(scale, w_f32.shape(), w_f32.device())?;
    let pos_mask = w_f32.ge(&threshold)?;
    let neg_mask = w_f32.le(&threshold.neg()?)?;
    let pos_f = pos_mask.to_dtype(DType::F32)?;
    let neg_f = neg_mask.to_dtype(DType::F32)?;
    let quantized = (pos_f - neg_f)?;
    Ok((quantized, scale))
}

/// Quantize activations to INT8 with per-tensor scaling.
/// scale = max(|x|) / 127
/// Returns (quantized: Tensor[f32 in -127..127], scale: f32)
pub fn quantize_int8(x: &Tensor) -> anyhow::Result<(Tensor, f32)> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let abs_x = x_f32.abs()?;
    let max_val = abs_x.max_all()?.to_scalar::<f32>()?;
    if max_val == 0.0 {
        return Ok((x_f32, 1.0));
    }
    let scale = max_val / 127.0;
    let scaled = (x_f32 / scale as f64)?;
    // Clamp to [-127, 127]
    let clamped = scaled.clamp(-127.0_f64, 127.0_f64)?;
    Ok((clamped, scale))
}

/// Dequantize: result * scale_w * scale_x / 127
pub fn dequantize(
    result: &Tensor,
    scale_w: f32,
    scale_x: f32,
) -> anyhow::Result<Tensor> {
    let factor = (scale_w * scale_x / 127.0) as f64;
    Ok((result.to_dtype(DType::F32)? * factor)?)
}

/// Pack two ternary values into one byte.
/// Maps: -1→0, 0→1, +1→2; packs two into a byte: high=val[2i], low=val[2i+1]
pub fn pack_ternary(weights: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(weights.len().div_ceil(2));
    let mut i = 0;
    while i < weights.len() {
        let hi = encode_ternary(weights[i]);
        let lo = if i + 1 < weights.len() { encode_ternary(weights[i + 1]) } else { 1 };
        packed.push((hi << 4) | lo);
        i += 2;
    }
    packed
}

fn encode_ternary(v: i8) -> u8 {
    match v {
        -1 => 0,
        0  => 1,
        1  => 2,
        _  => 1,
    }
}

fn decode_ternary(v: u8) -> i8 {
    match v {
        0 => -1,
        1 => 0,
        2 => 1,
        _ => 0,
    }
}

/// Unpack from packed ternary bytes back to i8 slice.
pub fn unpack_ternary(packed: &[u8], count: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    for &byte in packed {
        if out.len() < count {
            out.push(decode_ternary((byte >> 4) & 0x0F));
        }
        if out.len() < count {
            out.push(decode_ternary(byte & 0x0F));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip_even_and_odd() {
        let even = [-1i8, 0, 1, 1, 0, -1];
        assert_eq!(unpack_ternary(&pack_ternary(&even), even.len()), even);
        // Odd length: the dangling nibble is padded but `count` truncates it.
        let odd = [1i8, -1, 0];
        assert_eq!(unpack_ternary(&pack_ternary(&odd), odd.len()), odd);
    }

    #[test]
    fn quantize_dequantize_2d_roundtrip() {
        let dev = Device::Cpu;
        // mean(|w|) = (2+2+0+0.5+0.5+3)/6 = 1.3333 -> threshold.
        // ge:  2.0, 3.0 -> +1 ; le -thr: -2.0 -> -1 ; rest -> 0.
        let w = Tensor::from_vec(
            vec![2.0f32, -2.0, 0.0, 0.5, -0.5, 3.0],
            (2, 3),
            &dev,
        )
        .unwrap();

        let (packed, scale, rows, cols) = quantize_pack_2d(&w).unwrap();
        assert_eq!((rows, cols), (2, 3));
        assert!((scale - 8.0 / 6.0).abs() < 1e-5);

        let back = dequantize_unpack_2d(&packed, scale, rows, cols, &dev).unwrap();
        let got: Vec<f32> = back.flatten_all().unwrap().to_vec1().unwrap();
        let expected = vec![scale, -scale, 0.0, 0.0, 0.0, scale];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "got {got:?} expected {expected:?}");
        }
    }
}
