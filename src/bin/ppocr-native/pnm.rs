//! Binary 8-bit PGM and PPM, the uncompressed images the worker reads:
//! pdftoppm writes PPM, and `cv2.imwrite` writes either for a `.ppm` path.

use ppocr_native::Image;

/// Decode a P5 (grey) or P6 (RGB) file with a maximum value of 255, as
/// `cv2.imread` returns it in colour: BGR, grey copied to all three channels.
/// Bytes after the pixels are ignored, as OpenCV ignores them.
pub fn decode(mut bytes: Vec<u8>) -> Result<Image, String> {
    let mut at = 0;
    let mut fields = [0usize; 4];
    for (i, field) in fields.iter_mut().enumerate() {
        // Whitespace, and comments running to the end of their line.
        loop {
            match bytes.get(at) {
                Some(b) if b.is_ascii_whitespace() => at += 1,
                Some(b'#') => {
                    while bytes.get(at).is_some_and(|&b| b != b'\n') {
                        at += 1;
                    }
                }
                _ => break,
            }
        }
        let start = at;
        if i == 0 {
            at += 2;
            *field = match bytes.get(start..at) {
                Some(b"P5") => 1,
                Some(b"P6") => 3,
                _ => return Err("not a binary PGM or PPM".into()),
            };
            continue;
        }
        while bytes.get(at).is_some_and(u8::is_ascii_digit) {
            at += 1;
        }
        *field = std::str::from_utf8(&bytes[start..at])
            .ok()
            .and_then(|digits| digits.parse().ok())
            .ok_or("malformed header")?;
    }
    let [channels, width, height, max] = fields;
    if max != 255 {
        return Err(format!("{max} as maximum value, only 255 is read"));
    }
    if width == 0 || height == 0 {
        return Err("empty image".into());
    }
    // Exactly one whitespace byte separates the header from the pixels.
    if !bytes.get(at).is_some_and(u8::is_ascii_whitespace) {
        return Err("malformed header".into());
    }
    let start = at + 1;
    let pixels = width.checked_mul(height).ok_or("image too large")?;
    let len = pixels.checked_mul(channels).ok_or("image too large")?;
    if bytes.len() - start < len {
        return Err("truncated pixel data".into());
    }
    let data = if channels == 3 {
        bytes.truncate(start + len);
        bytes.drain(..start);
        for px in bytes.as_chunks_mut::<3>().0 {
            px.swap(0, 2);
        }
        bytes
    } else {
        let bgr_len = pixels.checked_mul(3).ok_or("image too large")?;
        let mut bgr = Vec::with_capacity(bgr_len);
        for &v in &bytes[start..start + len] {
            bgr.extend_from_slice(&[v, v, v]);
        }
        bgr
    };
    Ok(Image::new(width, height, data))
}

#[cfg(test)]
mod tests {
    use super::decode;

    #[test]
    fn reads_rgb_and_grey() {
        let ppm = b"P6\n# pdftoppm\n2 1\n255\n\x01\x02\x03\x04\x05\x06trailing".to_vec();
        let image = decode(ppm).unwrap();
        assert_eq!((image.width, image.height), (2, 1));
        assert_eq!(image.data, [3, 2, 1, 6, 5, 4]);
        let pgm = b"P5 1 2 255 \x07\x08".to_vec();
        assert_eq!(decode(pgm).unwrap().data, [7, 7, 7, 8, 8, 8]);
    }

    #[test]
    fn rejects_bad_headers() {
        for bad in [
            &b""[..],
            b"P3\n1 1\n255\n1 2 3",
            b"P6\n1 1\n65535\n\x00\x00\x00\x00\x00\x00",
            b"P6\n0 1\n255\n",
            b"P6\n2 2\n255\n\x00\x00\x00",
            b"P6\n1 1\n255",
            b"P6\n1 1\n255\n",
            b"P6\n18446744073709551615 3\n255\n\x00",
            b"P6\n99999999999999999999999 1\n255\n\x00",
            b"P6\n1 -1\n255\n\x00\x00\x00",
        ] {
            assert!(
                decode(bad.to_vec()).is_err(),
                "{:?}",
                String::from_utf8_lossy(bad)
            );
        }
    }
}
