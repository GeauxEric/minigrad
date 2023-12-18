#[derive(Debug)]
enum DType {
    F32(f32),
    U8(u8),
}
/// Holds the math data, derivative, operation, as well as some metadata, such as the label
#[derive(Debug)]
struct Value {
    data: DType
}

impl From<f32> for DType {
    fn from(value: f32) -> Self {
        DType::F32(value)
    }
}

impl From<u8> for DType {
    fn from(value: u8) -> Self {
        DType::U8(value)
    }
}

impl std::ops::Add for DType {
    type Output = DType;
    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (DType::U8(v1), DType::F32(v2)) => (v1 as f32 + v2).into(),
            (DType::U8(v1), DType::U8(v2)) => (v1 + v2).into(),
            (DType::F32(v1), DType::F32(v2)) => (v1  + v2).into(),
            (DType::F32(v1), DType::U8(v2)) => (v1  + v2 as f32).into(),

        }
    }
}

impl Value {
    pub fn new(data: impl Into<DType>) -> Self {
        Value { data: data.into() }
    }
}

/// Add operation
impl std::ops::Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        let d = self.data + rhs.data;
        Value::new(d)
    }
}


#[cfg(test)]
mod tests {
    use crate::Value;

    #[test]
    fn it_works() {
        let v1 = Value::new(1);
        let v2 = Value::new(1.0);
        println!("{:?} {:?}", v1, v2);
        let v3 = v1 + v2;
        println!("{:?}", v3);
    }
}
