use std::ops;

#[derive(Debug)]
enum DataType {
    U32(u32),
    F64(f64),
}

#[derive(Debug)]
struct Value {
    data: DataType,
}


impl Value {
    pub fn new(data: DataType) -> Self {
        Value { data }
    }
}

impl ops::Add for Value {
    type Output = DataType;

    fn add(self, rhs: Self) -> Self::Output {
        match (self.data, rhs.data) {
            (DataType::F64(f), DataType::U32(u)) => {
                DataType::F64(f + u as f64)
            }
            (DataType::F64(f), DataType::F64(u)) => {
                DataType::F64(f + u as f64)
            }
            (DataType::U32(f), DataType::U32(u)) => {
                DataType::U32(f + u)
            }
            (DataType::U32(f), DataType::F64(u)) => {
                DataType::F64(f as f64 + u)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let v1 = Value::new(DataType::U32(1));
        println!("{:?}", v1);
        let v2 = Value::new(DataType::F64(2.0));
        println!("{:?}", v1 + v2);
        // println!("{:?}", Value::new(1) * Value::new(2));
        // println!("{:?}", Value::new(1) * Value::new(2) + Value::new(3));
        // println!("{:?}", Value::new(1.0) * Value::new(2) + Value::new(3));
    }
}
