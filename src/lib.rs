/// Holds the math data, derivative, operation, as well as some metadata, such as the label
#[derive(Debug)]
struct Value<T> {
    data: T
}

impl<T> Value<T> {
    pub fn new(data: T) -> Self {
        Value{data}
    }
}

/// Add operation
impl<T> std::ops::Add for Value<T> {
    type Output = Value<T>;
    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}


#[cfg(test)]
mod tests {
    use crate::Value;

    #[test]
    fn it_works() {
        println!("{:?}", Value::new(1));
        let v1 = Value::new(1);
        let v2 = Value::new(1.0);
        let v3 = v1 + v2;
    }
}
