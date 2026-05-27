use memmap2::Mmap;
use rand::seq::SliceRandom;
use std::fs::File;
use std::path::Path;

/// Memory-mapped token dataset. Binary file of u32 token IDs, little-endian.
pub struct TokenDataset {
    mmap:       Mmap,
    pub seq_len:    usize,
    pub num_chunks: usize,
}

impl TokenDataset {
    pub fn open(path: &Path, seq_len: usize) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let total_tokens = mmap.len() / 4; // u32 = 4 bytes
        let num_chunks = total_tokens / seq_len;
        anyhow::ensure!(num_chunks > 0, "dataset too small for seq_len={seq_len}");
        Ok(Self { mmap, seq_len, num_chunks })
    }

    /// Get i-th chunk as u32 tokens.
    pub fn get(&self, idx: usize) -> anyhow::Result<Vec<u32>> {
        anyhow::ensure!(idx < self.num_chunks, "idx {idx} >= num_chunks {}", self.num_chunks);
        let start = idx * self.seq_len * 4;
        let end = start + self.seq_len * 4;
        let bytes = &self.mmap[start..end];
        Ok(bytes.chunks(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    /// Get (input, target) pair — target is input shifted left by 1.
    pub fn get_pair(&self, idx: usize) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
        let chunk = self.get(idx)?;
        let input: Vec<u32> = chunk[..chunk.len() - 1].to_vec();
        let target: Vec<u32> = chunk[1..].to_vec();
        Ok((input, target))
    }
}

pub struct DataLoader {
    dataset:    TokenDataset,
    batch_size: usize,
    shuffle:    bool,
    indices:    Vec<usize>,
    current:    usize,
}

impl DataLoader {
    pub fn new(dataset: TokenDataset, batch_size: usize, shuffle: bool) -> Self {
        let num = dataset.num_chunks;
        let mut indices: Vec<usize> = (0..num).collect();
        if shuffle {
            use rand::thread_rng;
            indices.shuffle(&mut thread_rng());
        }
        Self { dataset, batch_size, shuffle, indices, current: 0 }
    }

    /// Returns (input_ids, target_ids) both (B, T-1) as Vec<Vec<u32>>.
    pub fn next_batch(&mut self) -> anyhow::Result<Option<(Vec<Vec<u32>>, Vec<Vec<u32>>)>> {
        if self.current + self.batch_size > self.indices.len() {
            return Ok(None);
        }
        let mut inputs = Vec::with_capacity(self.batch_size);
        let mut targets = Vec::with_capacity(self.batch_size);
        for &idx in &self.indices[self.current..self.current + self.batch_size] {
            let (inp, tgt) = self.dataset.get_pair(idx)?;
            inputs.push(inp);
            targets.push(tgt);
        }
        self.current += self.batch_size;
        Ok(Some((inputs, targets)))
    }

    pub fn reset(&mut self) {
        self.current = 0;
        if self.shuffle {
            use rand::thread_rng;
            self.indices.shuffle(&mut thread_rng());
        }
    }

    pub fn num_batches(&self) -> usize {
        self.indices.len() / self.batch_size
    }
}
