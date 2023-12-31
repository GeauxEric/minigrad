use std::fmt::Formatter;
use std::ops;

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
struct Value {
    data: DataType,
    prev: Vec<Value>,
    op: Op,
    label: String,
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

impl Value {
    pub fn new(data: DataType) -> Self {
        Value {
            data,
            prev: vec![],
            op: Op::default(),
            label: "".to_string(),
            grad: 0f32,
        }
    }

    pub fn new_with_label(data: DataType, label: impl Into<String>) -> Self {
        let mut v = Self::new(data);
        v.with_label(label);
        v
    }

    pub fn with_child(&mut self, child: Value) -> &mut Self {
        self.prev.push(child);
        self
    }

    pub fn with_op(&mut self, op: Op) -> &mut Self {
        self.op = op;
        self
    }

    pub fn with_label(&mut self, label: impl Into<String>) -> &mut Self {
        self.label = label.into();
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
        let data = self.data.clone() + rhs.data.clone();
        let mut v = Value::new(data);
        v.with_child(self.clone())
            .with_child(rhs.clone())
            .with_op(Op::Plus);
        v
    }
}

impl ops::Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        let data = self.data.clone() * rhs.data.clone();
        let mut v = Value::new(data);
        v.with_child(self.clone())
            .with_child(rhs.clone())
            .with_op(Op::Mul);
        v
    }
}

#[cfg(test)]
mod tests {
    use graphviz_rust::{cmd::Format, exec, printer::PrinterContext};
    use graphviz_rust::cmd::CommandArg;
    use graphviz_rust::dot_generator::*;
    use graphviz_rust::dot_structures::*;

    use super::*;

    /// Plot the computation graph
    fn plot_computation_graph(value: &Value, graph: &mut Graph) {
        let id = &value.label;
        let n = node!(
            id,
            vec![
                attr!("label", esc format!("{} | data {} | grad {}", value.label, value.data, value.grad))
            ]
        );
        graph.add_stmt(n.into());
        if value.is_leaf() {
            return;
        }

        for child in &value.prev {
            // build the edge to the child value
            let child_id = &child.label;
            let e = edge!(node_id!(id) => node_id!(child_id), vec![attr!("label", esc format!("{}", value.op))]);
            graph.add_stmt(e.into());
            plot_computation_graph(child, graph)
        }
    }

    #[test]
    fn it_works() {
        let v1 = Value::new_with_label(DataType::F32(1.0), "v1");
        println!("{:?}", v1);
        let v2 = Value::new_with_label(DataType::F32(2.0), "v2");
        let mut v3 = v1 + v2;
        v3.with_label("v3");
        println!("{:?}", v3);
        let mut v5 = v3 * Value::new_with_label(DataType::F32(3.0), "v4");
        v5.with_label("v5");
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
