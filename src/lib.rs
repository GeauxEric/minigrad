use std::ops;

#[derive(Debug, Clone)]
enum DataType {
    U32(u32),
    F64(f64),
}

#[derive(Debug, Clone)]
struct Value {
    data: DataType,
    prev: Vec<Value>
}


impl Value {
    pub fn new(data: DataType) -> Self {
        Value { data , prev: vec![]}
    }

    pub fn add_child(&mut self, child: Value) {
        self.prev.push(child)
    }
}

impl ops::Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        let data = match (self.data.clone(), rhs.data.clone()) {
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
        };
        let mut v = Value::new(data);
        v.add_child(self.clone());
        v.add_child(rhs.clone());
        v
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
