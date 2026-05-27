/// Pack a stream of token IDs into fixed-length training chunks.
/// Each chunk is exactly `seq_len` tokens.
/// Documents separated by EOS token; chunks may span document boundaries.
pub fn pack_tokens(token_stream: &[u32], seq_len: usize, _eos_id: u32) -> Vec<Vec<u32>> {
    token_stream.chunks(seq_len)
        .filter(|c| c.len() == seq_len)
        .map(|c| c.to_vec())
        .collect()
}
