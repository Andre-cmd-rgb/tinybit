use tinybit_core::quantize::{pack_ternary, quantize_int8, quantize_ternary, unpack_ternary};
use candle_core::{DType, Device, Tensor};

#[test]
fn test_quantize_ternary_values() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let w = Tensor::from_vec(vec![1.0f32, -1.0, 0.0, 2.0, -0.5, 0.5], (2, 3), &device)?;
    let (q, scale) = quantize_ternary(&w)?;
    assert!(scale > 0.0, "scale should be positive");
    let q_vals = q.to_vec2::<f32>()?;
    for row in &q_vals {
        for &v in row {
            assert!(
                v == -1.0 || v == 0.0 || v == 1.0,
                "ternary value out of range: {v}"
            );
        }
    }
    Ok(())
}

#[test]
fn test_quantize_int8_range() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let x = Tensor::from_vec(vec![0.0f32, 1.0, -1.0, 0.5, 100.0, -50.0], 6, &device)?;
    let (q, scale) = quantize_int8(&x)?;
    assert!(scale > 0.0);
    let q_vals = q.to_vec1::<f32>()?;
    for &v in &q_vals {
        assert!(v >= -127.0 && v <= 127.0, "int8 value out of range: {v}");
    }
    Ok(())
}

#[test]
fn test_pack_unpack_roundtrip() {
    let original: Vec<i8> = vec![-1, 0, 1, -1, 1, 0, 1, -1, 0];
    let packed = pack_ternary(&original);
    let unpacked = unpack_ternary(&packed, original.len());
    assert_eq!(unpacked, original, "pack/unpack roundtrip failed");
}

#[test]
fn test_pack_even_count() {
    let vals: Vec<i8> = vec![1, -1, 0, 1];
    let packed = pack_ternary(&vals);
    assert_eq!(packed.len(), 2); // 4 values → 2 bytes
    let unpacked = unpack_ternary(&packed, 4);
    assert_eq!(unpacked, vals);
}

#[test]
fn test_pack_odd_count() {
    let vals: Vec<i8> = vec![1, -1, 0];
    let packed = pack_ternary(&vals);
    assert_eq!(packed.len(), 2); // 3 values → 2 bytes (last padded)
    let unpacked = unpack_ternary(&packed, 3);
    assert_eq!(unpacked, vals);
}
