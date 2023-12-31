use std::fmt::Formatter;
use std::ops;
use std::rc::Rc;

#[derive(Debug, Clone)]
enum DataType {
    F32(f32),
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::F32(v) => v.fmt(f),
        }
    }
}

impl ops::Add for DataType {
    type Output = DataType;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (DataType::F32(v1), DataType::F32(v2)) => DataType::F32(v1 + v2),
        }
    }
}

impl ops::Mul for DataType {
    type Output = DataType;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (DataType::F32(v1), DataType::F32(v2)) => DataType::F32(v1 * v2),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
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
struct ValueId(usize);

impl ValueId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

impl std::fmt::Display for ValueId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone)]
struct Value_ {
    data: DataType,
    prev: Vec<Value>,
    op: Op,
    id: ValueId,
    grad: f32,
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::NoOp => {
                write!(f, "{}", "no-op")
            }
            Op::Plus => {
                write!(f, "{}", "+")
            }
            Op::Mul => {
                write!(f, "{}", "*")
            }
        }
    }
}

/// TODO: cyclic dependency can lead to memory leak
#[derive(Debug)]
struct Value(Rc<Value_>);

impl Clone for Value {
    fn clone(&self) -> Self {
        Value(self.0.clone())
    }
}

impl std::ops::Deref for Value {
    type Target = Value_;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl Value_ {
    pub fn new(data: DataType) -> Self {
        Value_ {
            data,
            prev: vec![],
            op: Op::default(),
            id: ValueId::new(),
            grad: 0f32,
        }
    }

    pub fn with_child(&mut self, child: Value) -> &mut Self {
        self.prev.push(child);
        self
    }

    pub fn with_op(&mut self, op: Op) -> &mut Self {
        self.op = op;
        self
    }

    pub fn with_grad(&mut self, grad: f32) -> &mut Self {
        self.grad = grad;
        self
    }

    pub fn is_leaf(&self) -> bool {
        return self.op == Op::NoOp;
    }
}

impl ops::Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        let data = self.0.data.clone() + rhs.0.data.clone();
        let mut v = Value_::new(data);
        v.with_child(self.clone())
            .with_child(rhs.clone())
            .with_op(Op::Plus);
        Value(Rc::new(v))
    }
}

impl ops::Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        let data = self.0.data.clone() * rhs.0.data.clone();
        let mut v = Value_::new(data);
        v.with_child(self.clone())
            .with_child(rhs.clone())
            .with_op(Op::Mul);
        Value(Rc::new(v))
    }
}

#[cfg(test)]
mod tests {
    use graphviz_rust::cmd::CommandArg;
    use graphviz_rust::dot_generator::*;
    use graphviz_rust::dot_structures::*;
    use graphviz_rust::{cmd::Format, exec, printer::PrinterContext};

    use super::*;

    /// Plot the computation graph
    fn plot_computation_graph(value: &Value_, graph: &mut Graph) {
        let id = &value.id;
        let n = node!(
            id,
            vec![
                attr!("label", esc format!("{} | data {} | grad {}", value.id, value.data, value.grad))
            ]
        );
        graph.add_stmt(n.into());
        if value.is_leaf() {
            return;
        }

        for child in &value.prev {
            // build the edge to the child value
            let child_id = &child.id;
            let e = edge!(node_id!(id) => node_id!(child_id), vec![attr!("label", esc format!("{}", value.op))]);
            graph.add_stmt(e.into());
            plot_computation_graph(child, graph)
        }
    }

    #[test]
    fn it_works() {
        let v1 = Value(Rc::new(Value_::new(DataType::F32(1.0))));
        let v2 = Value_::new(DataType::F32(2.0));
        let v2 = Value(Rc::new(v2));
        let v3 = v1 + v2;
        println!("{:?}", v3);
        let v4 = Value_::new(DataType::F32(3.0));
        let v4 = Value(Rc::new(v4));
        let v5 = v3 * v4;
        println!("{:?}", v5);

        let mut g = graph!(id!("computation"));
        plot_computation_graph(&v5, &mut g);
        let _graph_svg = exec(
            g,
            &mut PrinterContext::default(),
            vec![Format::Svg.into(), CommandArg::Output("1.svg".to_string())],
        )
        .unwrap();
    }
}
