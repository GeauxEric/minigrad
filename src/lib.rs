#[derive(Debug, Clone)]
enum DType {
    F32(f32),
    U8(u8),
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
            (DType::F32(v1), DType::F32(v2)) => (v1 + v2).into(),
            (DType::F32(v1), DType::U8(v2)) => (v1 + v2 as f32).into(),
        }
    }
}

impl std::ops::Mul for DType {
    type Output = DType;
    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (DType::U8(v1), DType::F32(v2)) => (v1 as f32 * v2).into(),
            (DType::U8(v1), DType::U8(v2)) => (v1 * v2).into(),
            (DType::F32(v1), DType::F32(v2)) => (v1 * v2).into(),
            (DType::F32(v1), DType::U8(v2)) => (v1 * v2 as f32).into(),
        }
    }
}

#[derive(Debug, Clone)]
enum Op {
    NoOp,
    Plus,
    Mul,
}

/// Holds the math data, derivative, operation, as well as some metadata, such as the label
#[derive(Debug, Clone)]
struct Value {
    /// current data
    data: DType,

    /// uid
    label: String,

    /// math operation that produces the data
    op: Op,

    /// values that were used to produce this value
    /// TODO: avoid clone
    prev: Vec<Value>,
}

impl Value {
    pub fn new(data: impl Into<DType>) -> Self {
        Value {
            data: data.into(),
            label: "".into(),
            op: Op::NoOp,
            prev: vec![],
        }
    }

    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = label.into();
    }
}

/// Add operation
impl std::ops::Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        let d = self.clone().data + rhs.clone().data;
        let mut v = Value::new(d);
        v.op = Op::Plus;
        v.prev.push(self);
        v.prev.push(rhs);
        v
    }
}

/// Mul
impl std::ops::Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        let d = self.clone().data * rhs.clone().data;
        let mut v = Value::new(d);
        v.op = Op::Mul;
        v.prev.push(self);
        v.prev.push(rhs);
        v
    }
}

#[cfg(test)]
mod tests {
    use crate::Value;

    #[test]
    fn it_works() {
        let mut v1 = Value::new(1);
        v1.set_label("v1");
        let mut v2 = Value::new(1.0);
        v2.set_label("v2");
        println!("{:?} {:?}", v1, v2);
        let mut v3 = v1 + v2;
        v3.set_label("v3");
        println!("{:?}", v3);
        let mut v4 = Value::new(3);
        v4.set_label("v4");
        let mut v5 = v4 * v3;
        v5.set_label("v5");
        println!("{:?}", v5);
    }
}
