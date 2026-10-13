//! Just enough of numpy's `.npy` format to read the reference arrays:
//! little-endian, C order, version 1 to 3 headers.

use std::fs;
use std::path::Path;

use ppocr_native::Image;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// An array's numpy type string (`<f4`, `|u1`, ...), shape and raw bytes.
pub struct Npy {
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Npy {
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if !bytes.starts_with(b"\x93NUMPY") {
            return Err(format!("{}: not a .npy file", path.display()).into());
        }
        let (len, start) = if bytes[6] == 1 {
            (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10)
        } else {
            (u32::from_le_bytes(bytes[8..12].try_into()?) as usize, 12)
        };
        let header = std::str::from_utf8(&bytes[start..start + len])?;
        let field = |key: &str| -> Result<&str> {
            let at = header
                .find(key)
                .ok_or_else(|| format!("{}: no {key}", path.display()))?
                + key.len();
            Ok(header[at..].trim_start_matches([':', ' ', '\'']))
        };
        if field("'fortran_order'")?.starts_with("True") {
            return Err(format!("{}: Fortran order", path.display()).into());
        }
        let descr = field("'descr'")?
            .split('\'')
            .next()
            .unwrap_or_default()
            .to_owned();
        let dims = field("'shape'")?;
        let dims = &dims[1..dims.find(')').ok_or("shape")?];
        let shape = dims
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::parse)
            .collect::<std::result::Result<Vec<usize>, _>>()?;
        Ok(Self {
            descr,
            shape,
            data: bytes[start + len..].to_vec(),
        })
    }

    fn expect(&self, descr: &str) -> Result<()> {
        if self.descr == descr {
            Ok(())
        } else {
            Err(format!("expected {descr}, got {}", self.descr).into())
        }
    }

    pub fn f32s(&self) -> Result<Vec<f32>> {
        self.expect("<f4")?;
        Ok(self
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    pub fn i32s(&self) -> Result<Vec<i32>> {
        self.expect("<i4")?;
        Ok(self
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect())
    }

    pub fn image(&self) -> Result<Image> {
        self.expect("|u1")?;
        match self.shape[..] {
            [h, w, 3] => Ok(Image::new(w, h, self.data.clone())),
            _ => Err(format!("not an HxWx3 image: {:?}", self.shape).into()),
        }
    }
}
