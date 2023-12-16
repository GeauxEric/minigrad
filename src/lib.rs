use std::ops;

#[derive(Debug)]
struct Value<T> {
    data: T,
}


impl<T> Value<T> {
    pub fn new(data: T) -> Self  {
        Value { data }
    }
}

impl<T> ops::Add for Value<T>
where T: ops::Add<Output = T> {
    type Output = Value<T> ;

    fn add(self, rhs: Self) -> Self::Output {
        Value::new(self.data + rhs.data)
    }
}


impl<T> ops::Mul for Value<T>
    where T: ops::Mul<Output = T> {
    type Output = Value<T> ;

    fn mul(self, rhs: Self) -> Self::Output {
        Value::new(self.data * rhs.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let v1 = Value::new(1);
        println!("{:?}", v1);
        let v2 = Value::new(2);
        println!("{:?}", v1 + v2);
        println!("{:?}", Value::new(1) * Value::new(2));
        println!("{:?}", Value::new(1) * Value::new(2) + Value::new(3));
    }
}
