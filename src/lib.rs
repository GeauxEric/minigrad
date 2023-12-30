use std::fmt::Formatter;

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

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32(v) => {
                write!(f, "{}", v)
            }
            DType::U8(v) => {
                write!(f, "{}", v)
            }
        }
    }
}

macro_rules! data_type_op_impl {
    ($func:ident, $bound:ident, $op:tt) => {
        impl std::ops::$bound for DType {
            type Output = DType;
            fn $func(self, rhs: Self) -> Self::Output {
                match (self, rhs) {
                    (DType::U8(v1), DType::F32(v2)) => (v1 as f32 $op v2).into(),
                    (DType::U8(v1), DType::U8(v2)) => (v1 $op v2).into(),
                    (DType::F32(v1), DType::F32(v2)) => (v1 $op v2).into(),
                    (DType::F32(v1), DType::U8(v2)) => (v1 $op v2 as f32).into(),
                }
            }
        }
    };
}

data_type_op_impl!(sub, Sub, -);
data_type_op_impl!(add, Add, +);
data_type_op_impl!(mul, Mul, *);
data_type_op_impl!(div, Div, /);

#[derive(Debug, Clone, PartialEq)]
enum Op {
    None,
    Plus,
    Mul,
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::None => {
                write!(f, "")
            }
            Op::Plus => {
                write!(f, "+")
            }
            Op::Mul => {
                write!(f, "*")
            }
        }
    }
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

struct ValueBuilder(Value);

impl ValueBuilder {
    pub fn new(data: impl Into<DType>) -> Self {
        ValueBuilder(Value {
            data: data.into(),
            label: "".to_string(),
            op: Op::None,
            prev: vec![],
        })
    }

    pub fn label(&mut self, label: &str) -> &mut Self {
        self.0.label = label.to_string();
        self
    }

    pub fn op(&mut self, op: Op) -> &mut Self {
        self.0.op = op;
        self
    }

    pub fn add_prev(&mut self, prev: Value) -> &mut Self {
        self.0.prev.push(prev);
        self
    }

    pub fn build(self) -> Value {
        self.0
    }
}

impl Value {
    pub fn is_leaf(&self) -> bool {
        self.op == Op::None
    }

    pub fn set_label(&mut self, label: &str) {
        self.label = label.to_string()
    }
}

/// Add operation
impl std::ops::Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        let d = self.clone().data + rhs.clone().data;
        let mut v = ValueBuilder::new(d);
        v.op(Op::Plus).add_prev(self).add_prev(rhs);
        v.build()
    }
}

/// Mul
impl std::ops::Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        let d = self.clone().data * rhs.clone().data;
        let mut v = ValueBuilder::new(d);
        v.op(Op::Mul).add_prev(self).add_prev(rhs);
        v.build()
    }
}

#[cfg(test)]
mod tests {
    use graphviz_rust::cmd::CommandArg::Output;
    use graphviz_rust::dot_generator::*;
    use graphviz_rust::dot_structures::*;
    use graphviz_rust::{cmd::Format, exec, printer::PrinterContext};

    use crate::{Value, ValueBuilder};

    fn viz_computation_graph(value: &Value, graph: &mut Graph) {
        let value_node_id = value.label.clone();
        let value_node = node!(
            value_node_id,
            vec![attr!("label", esc format!("{}", value.data))]
        );
        graph.add_stmt(value_node.into());
        // if value is the leaf, add node to graph and return
        if value.is_leaf() {
            return;
        }
        // otherwise, recursively add to the graph
        for p in &value.prev {
            let e = edge!(node_id!(p.label) => node_id!(value_node_id), vec![attr!("label", esc format!("{}", value.op))]);
            graph.add_stmt(e.into());
            viz_computation_graph(p, graph);
        }
    }

    #[test]
    fn it_works() {
        let mut v1 = ValueBuilder::new(1);
        v1.label("v1");
        let v1 = v1.build();
        let mut v2 = ValueBuilder::new(1.0);
        v2.label("v2");
        let v2 = v2.build();
        println!("{:?} {:?}", v1, v2);
        let mut v3 = v1 + v2;
        v3.set_label("v3");
        println!("{:?}", v3);
        let mut v4 = ValueBuilder::new(3);
        v4.label("v4");
        let v4 = v4.build();
        let mut v5 = v4 * v3;
        v5.set_label("v5");
        println!("{:?}", v5);

        let mut g = graph!(id!("computation"));
        viz_computation_graph(&v5, &mut g);
        let _graph_svg = exec(
            g,
            &mut PrinterContext::default(),
            vec![Format::Svg.into(), Output("./1.svg".into())],
        )
        .unwrap();
    }
}
