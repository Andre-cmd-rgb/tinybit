use candle_core::{Device, DType, Tensor};

/// Marker tensor name present in a tinybit quantized-export safetensors file.
/// `TinyBit::load` detects it and reconstructs full-precision weights.
pub const QUANT_MARKER: &str = "__tinybit_quant__";

/// Quantize a 2D weight matrix to packed ternary for export.
/// Returns (packed_bytes, scale, rows, cols). Five ternary values are packed
/// per byte (base-3: 3^5 = 243 ≤ 256) → ~1.6 bits/weight, i.e. the packed
/// payload is ~1/20 the size of the original f32 matrix.
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
    // BitNet b1.58 absmean ternarization: W̃ = round(clip(W/scale, -1, +1)) with
    // scale = mean(|W|). round() sends |W| ≥ 0.5·scale to ±1 and everything
    // smaller to 0. (The earlier cutoff used the full `scale`, which zeroed
    // ~twice as many weights as b1.58 prescribes — a much sparser, lossier
    // export. `scale` is still the reconstruction magnitude.)
    let cutoff = 0.5 * scale;
    let threshold = Tensor::full(cutoff, w_f32.shape(), w_f32.device())?;
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

/// Pack ternary values into bytes, FIVE per byte using base-3 (3^5 = 243 ≤ 256).
/// Each value is encoded -1→0, 0→1, +1→2, then a group of five trits is combined
/// low-order-first: t0 + 3·t1 + 9·t2 + 27·t3 + 81·t4. This is ~1.6 bits/weight —
/// 2.5× denser than the old 4-bit nibble packing.
pub fn pack_ternary(weights: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(weights.len().div_ceil(5));
    for chunk in weights.chunks(5) {
        let mut byte: u16 = 0;
        let mut mult: u16 = 1;
        for &w in chunk {
            byte += encode_ternary(w) as u16 * mult;
            mult *= 3;
        }
        // A short final chunk leaves the high-order trits at 0 (which decode to
        // -1), but those sit beyond `count` on unpack and are truncated away.
        packed.push(byte as u8);
    }
    packed
}

fn encode_ternary(v: i8) -> u8 {
    match v {
        -1 => 0,
        0 => 1,
        1 => 2,
        _ => 1,
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

/// Unpack base-3-packed ternary bytes (five trits/byte, low-order first) back to
/// an i8 slice, truncated to `count`. Inverse of [`pack_ternary`].
pub fn unpack_ternary(packed: &[u8], count: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    for &byte in packed {
        let mut v = byte as u16;
        for _ in 0..5 {
            if out.len() >= count {
                return out;
            }
            out.push(decode_ternary((v % 3) as u8));
            v /= 3;
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
        // Length not a multiple of 5: the dangling trits are padded but `count`
        // truncates them.
        let odd = [1i8, -1, 0];
        assert_eq!(unpack_ternary(&pack_ternary(&odd), odd.len()), odd);
        // Longer sequence spanning several base-3 bytes, length not a multiple of 5.
        let long = [1i8, -1, 0, 1, 1, -1, 0, 0, 1, -1, 1, 0, -1];
        assert_eq!(long.len(), 13);
        assert_eq!(unpack_ternary(&pack_ternary(&long), long.len()), long);
        // Density: five trits fit in exactly one byte (3^5 = 243 ≤ 256).
        assert_eq!(pack_ternary(&[1i8, 1, 1, 1, 1]).len(), 1);
        assert_eq!(pack_ternary(&[1i8, 1, 1, 1, 1, -1]).len(), 2);
    }

    #[test]
    fn quantize_dequantize_2d_roundtrip() {
        let dev = Device::Cpu;
        // scale = mean(|w|) = (2+2+0+0.5+0.5+3)/6 = 1.3333; b1.58 cutoff = 0.5·scale
        // = 0.667. |w| >= 0.667 -> ±1 (2.0, 3.0 -> +1; -2.0 -> -1); 0.5/-0.5/0 -> 0.
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
