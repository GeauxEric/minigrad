use std::collections::HashMap;
use std::fmt::Formatter;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    id: usize,

    /// math operation that produces the data
    op: Op,

    /// values that were used to produce this value
    /// TODO: avoid clone
    prev: Vec<Value>,

    /// derivative of root value w.r.t. this value
    grad: f32,
}

/// Calculate grad from root value
fn calculate_grad(root: &mut Value) {
    root.set_grad(1.0);

    if root.is_leaf() {
        return;
    }
    let mut prev = HashMap::new();
    root.prev.iter().for_each(|v| {
        prev.insert(v.id.clone(), v.data);
    });
    root.prev.iter_mut().for_each(|v| {
        let mut my_prev = prev.clone();
        my_prev.remove(&v.id);
        let sibling_data: Vec<f32> = my_prev.values().cloned().collect();
        calculate_grad_non_root(v, root.grad, root.op.clone(), &sibling_data)
    })
}

fn calculate_grad_non_root(
    value: &mut Value,
    parent_grad: f32,
    parent_op: Op,
    sibling_data: &Vec<f32>,
) {
    match parent_op {
        Op::NoOp => {
            panic!("should not reach here! Calculate grad of prev of leaf node.")
        }
        Op::Plus => {
            // v = v1 + v2
            // d(v) / d(v1) = 1
            // d(L) / d(v1) = d(L) / d(v) * d(v) / d(v1) = parent_grad * 1.0
            let local_grad = 1.0;
            let grad = parent_grad * local_grad;
            value.set_grad(grad);
            if value.is_leaf() {
                return;
            }
            let mut prev = HashMap::new();
            value.prev.iter().for_each(|v| {
                prev.insert(v.id.clone(), v.data);
            });
            value.prev.iter_mut().for_each(|v| {
                let mut my_prev = prev.clone();
                my_prev.remove(&v.id);
                let sibling_data: Vec<f32> = my_prev.values().cloned().collect();
                calculate_grad_non_root(v, value.grad, value.op.clone(), &sibling_data)
            })
        }
        Op::Mul => {
            // v = v1 * v2
            // d(v) / d(v1) = v2
            // d(L) / d(v1) = d(L) / d(v) * d(v) / d(v1) = parent_grad * v2
            assert_eq!(sibling_data.len(), 1);
            let v2 = sibling_data[0];
            let grad = parent_grad * v2;
            value.set_grad(grad);
            if value.is_leaf() {
                return;
            }
            let mut prev = HashMap::new();
            value.prev.iter().for_each(|v| {
                prev.insert(v.id.clone(), v.data);
            });
            value.prev.iter_mut().for_each(|v| {
                let mut my_prev = prev.clone();
                my_prev.remove(&v.id);
                let sibling_data: Vec<f32> = my_prev.values().cloned().collect();
                calculate_grad_non_root(v, value.grad, value.op.clone(), &sibling_data)
            })
        }
    }
}

fn get_id() -> usize {
    static COUNTER: AtomicUsize = AtomicUsize::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

impl Value {
    pub fn new(data: f32) -> Self {
        let id = get_id();
        Value {
            data,
            id,
            op: Op::NoOp,
            prev: vec![],
            grad: 0.0,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.op == Op::NoOp
    }

    pub fn set_grad(&mut self, grad: f32) {
        self.grad = grad;
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

    use crate::{calculate_grad, Value};

    fn viz_computation_graph(value: &Value, graph: &mut Graph) {
        let value_node_id = value.id;
        let value_node = node!(
            value_node_id,
            vec![
                attr!("label", esc format!("{} | data={} grad={}", value.id, value.data, value.grad))
            ]
        );
        graph.add_stmt(value_node.into());
        // if value is the leaf, add node to graph and return
        if value.is_leaf() {
            return;
        }
        // otherwise, recursively add to the graph
        for p in &value.prev {
            let p_node_id = p.id;
            let e = edge!(node_id!(p_node_id) => node_id!(value_node_id), vec![attr!("label", esc format!("{}", value.op))]);
            graph.add_stmt(e.into());
            viz_computation_graph(p, graph);
        }
    }

    #[test]
    fn it_works() {
        let v1 = Value::new(3.0);
        let v2 = Value::new(2.0);
        println!("{:?} {:?}", v1, v2);
        let v3 = v1 * v2;
        println!("{:?}", v3);
        let v4 = Value::new(1.0);
        let mut v5 = v4 + v3;
        println!("{:?}", v5);

        calculate_grad(&mut v5);

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
