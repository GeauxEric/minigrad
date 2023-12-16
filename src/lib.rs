use std::ops;

#[derive(Debug, Clone)]
enum DataType {
    U32(u32),
    F64(f64),
}

#[derive(Debug, Clone)]
enum Op {
    NoOp,
    Plus,
    Mul,
}

impl Default for Op {
    fn default() -> Self {
        Op::NoOp
    }
}

#[derive(Debug, Clone)]
struct Value {
    data: DataType,
    prev: Vec<Value>,
    op: Op,
    label: String,
}


impl Value {
    pub fn new(data: DataType) -> Self {
        Value { data, prev: vec![], op: Op::default(), label: "".to_string() }
    }

    pub fn new_with_label(data: DataType, label: impl Into<String>) -> Self {
        let mut v = Self::new(data);
        v.with_label(label);
        v
    }

    pub fn with_child(&mut self, child: Value) {
        self.prev.push(child)
    }

    pub fn with_op(&mut self, op: Op) {
        self.op = op;
    }

    pub fn with_label(&mut self, label: impl Into<String>) {
        self.label = label.into();
    }
}

impl ops::Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        let data = match (self.data.clone(), rhs.data.clone()) {
            (DataType::F64(f), DataType::U32(u)) => {
                DataType::F64(f + u as f64)
            }
            (DataType::F64(f), DataType::F64(f2)) => {
                DataType::F64(f + f2)
            }
            (DataType::U32(f), DataType::U32(u)) => {
                DataType::U32(f + u)
            }
            (DataType::U32(u1), DataType::F64(f)) => {
                DataType::F64(u1 as f64 + f)
            }
        };
        let mut v = Value::new(data);
        v.with_child(self.clone());
        v.with_child(rhs.clone());
        v.with_op(Op::Plus);
        v
    }
}

impl ops::Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        let data = match (self.data.clone(), rhs.data.clone()) {
            (DataType::F64(f), DataType::U32(u)) => {
                DataType::F64(f * u as f64)
            }
            (DataType::F64(f), DataType::F64(f2)) => {
                DataType::F64(f * f2)
            }
            (DataType::U32(f), DataType::U32(u)) => {
                DataType::U32(f * u)
            }
            (DataType::U32(u1), DataType::F64(f)) => {
                DataType::F64(u1 as f64 * f)
            }
        };
        let mut v = Value::new(data);
        v.with_child(self.clone());
        v.with_child(rhs.clone());
        v.with_op(Op::Mul);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let v1 = Value::new_with_label(DataType::U32(1), "v1");
        println!("{:?}", v1);
        let v2 = Value::new_with_label(DataType::F64(2.0), "v2");
        let mut v3 = v1 + v2;
        v3.with_label("v3");
        println!("{:?}", v3);
        let mut v5 = v3 * Value::new_with_label(DataType::F64(3.0), "v4");
        v5.with_label("v5");
        println!("{:?}", v5);
    }
}
