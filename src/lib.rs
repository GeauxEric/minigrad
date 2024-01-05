use std::fmt::Formatter;

#[derive(Debug, Clone, PartialEq)]
enum Op {
    NoOp,
    Plus,
    Mul,
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::NoOp => {
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
    data: f32,

    /// uid
    label: String,

    /// math operation that produces the data
    op: Op,

    /// values that were used to produce this value
    /// TODO: avoid clone
    prev: Vec<Value>,
}

impl Value {
    pub fn new(data: f32) -> Self {
        Value {
            data,
            label: "".into(),
            op: Op::NoOp,
            prev: vec![],
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.op == Op::NoOp
    }

    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = label.into();
    }
}

/// Add operation
impl std::ops::Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        let d = self.data + rhs.data;
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
        let d = self.data * rhs.data;
        let mut v = Value::new(d);
        v.op = Op::Mul;
        v.prev.push(self);
        v.prev.push(rhs);
        v
    }
}

#[cfg(test)]
mod tests {
    use graphviz_rust::{cmd::Format, exec, printer::PrinterContext};
    use graphviz_rust::cmd::CommandArg::Output;
    use graphviz_rust::dot_generator::*;
    use graphviz_rust::dot_structures::*;

    use crate::Value;

    fn viz_computation_graph(value: &Value, graph: &mut Graph) {
        let value_node_id = value.label.clone();
        let value_node = node!(
            value_node_id,
            vec![attr!("label", esc format!("{} | data={}", value.label, value.data))]
        );
        graph.add_stmt(value_node.into());
        // if value is the leaf, add node to graph and return
        if value.is_leaf() {
            return;
        }
        // otherwise, recursively add to the graph
        for p in &value.prev {
            let p_node_id = p.label.clone();
            let e = edge!(node_id!(p_node_id) => node_id!(value_node_id), vec![attr!("label", esc format!("{}", value.op))]);
            graph.add_stmt(e.into());
            viz_computation_graph(p, graph);
        }
    }

    #[test]
    fn it_works() {
        let mut v1 = Value::new(2.0);
        v1.set_label("v1");
        let mut v2 = Value::new(2.0);
        v2.set_label("v2");
        println!("{:?} {:?}", v1, v2);
        let mut v3 = v1 * v2;
        v3.set_label("v3");
        println!("{:?}", v3);
        let mut v4 = Value::new(1.0);
        v4.set_label("v4");
        let mut v5 = v4 + v3;
        v5.set_label("L");
        println!("{:?}", v5);

        let mut g = graph!(id!("computation"));
        viz_computation_graph(&v5, &mut g);
        let _graph_svg = exec(
            g,
            &mut PrinterContext::default(),
            vec![Format::Png.into(), Output("./1.png".into())],
        )
        .unwrap();
    }
}
