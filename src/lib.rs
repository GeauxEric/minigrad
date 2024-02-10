use std::cell::RefCell;
use std::fmt::Formatter;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone)]
enum Op {
    None,
    Plus(Value, Value),
    Mul(Value, Value),
    Tanh(Value),
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::None => {
                write!(f, "")
            }
            Op::Plus(_, _) => {
                write!(f, "+")
            }
            Op::Mul(_, _) => {
                write!(f, "*")
            }
            Op::Tanh(_) => {
                write!(f, "tanh")
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Value(Rc<Value_>);

/// Holds the math data, derivative, operation, as well as some metadata, such as the label
#[derive(Debug)]
struct Value_ {
    /// current data
    data: RefCell<f32>,

    /// uid
    id: usize,

    /// math operation that produces the data
    op: Op,

    /// derivative of root value w.r.t. this value
    grad: RefCell<f32>,

    /// if true, the data shall be updated during the optimization process
    is_variable: bool,
}

impl Deref for Value {
    type Target = Value_;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Calculate grad from root value
fn calculate_grad(root: &Value) {
    *root.0.grad.borrow_mut() = 1.0;
    calculate_operand_grad(root);
}

fn calculate_operand_grad(value: &Value) {
    match &value.op {
        Op::None => {}
        Op::Plus(v1, v2) => {
            calculate_non_root_grad(v1, *value.grad.borrow(), &value.op);
            calculate_non_root_grad(v2, *value.grad.borrow(), &value.op);
        }
        Op::Mul(v1, v2) => {
            calculate_non_root_grad(v1, *value.grad.borrow(), &value.op);
            calculate_non_root_grad(v2, *value.grad.borrow(), &value.op);
        }
        Op::Tanh(v1) => {
            calculate_non_root_grad(v1, *value.grad.borrow(), &value.op);
        }
    }
}

fn calculate_non_root_grad(value: &Value, parent_grad: f32, parent_op: &Op) {
    // no usage of grad for non-variable
    if !value.is_variable {
        return;
    }
    match parent_op {
        Op::None => {
            panic!("should not reach here! Calculate grad of prev of leaf node.")
        }
        Op::Plus(_, _) => {
            // v = v1 + v2
            // d(v) / d(v1) = 1
            // d(L) / d(v1) = d(L) / d(v) * d(v) / d(v1) = parent_grad * 1.0
            let local_grad = 1.0;
            let grad = parent_grad * local_grad;
            *value.0.grad.borrow_mut() += grad;
            calculate_operand_grad(value);
        }
        Op::Mul(v1, v2) => {
            // v = v1 * v2
            // d(v) / d(v1) = v2
            // d(L) / d(v1) = d(L) / d(v) * d(v) / d(v1) = parent_grad * v2
            let d = if v1.id == value.id {
                v2.get_data()
            } else {
                v1.get_data()
            };
            let grad = parent_grad * d;
            *value.0.grad.borrow_mut() += grad;
            calculate_operand_grad(value)
        }
        Op::Tanh(v1) => {
            // v = tanh(v1)
            // d(v) / d(v1) = 1 - (tanh(v1)) ^ 2
            // d(L) / d(v1) = parent_grad * (1 - (tanh(v1)) ^ 2)
            let d = v1.get_data();
            let local_grad = 1.0 - d.tanh().powi(2);
            let grad = parent_grad * local_grad;
            *value.0.grad.borrow_mut() += grad;
            calculate_operand_grad(v1);
        }
    }
}

fn get_id() -> usize {
    static COUNTER: AtomicUsize = AtomicUsize::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

impl Value {
    pub fn new(data: f32) -> Self {
        Value(Rc::new(Value_::new(data, false)))
    }

    pub fn tanh(&self) -> Self {
        let d = *self.data.borrow();
        let t = ((2.0 * d).exp() - 1.0) / ((2.0 * d).exp() + 1.0);
        let mut v = Value_::new(t, true);
        v.op = Op::Tanh(self.clone());
        Value(Rc::new(v))
    }
}

impl Value_ {
    pub fn new(data: f32, is_variable: bool) -> Self {
        let id = get_id();
        Value_ {
            data: RefCell::new(data),
            id,
            is_variable,
            op: Op::None,
            grad: RefCell::new(0.0),
        }
    }

    pub fn get_data(&self) -> f32 {
        return *self.data.borrow();
    }

    pub fn get_grad(&self) -> f32 {
        return *self.grad.borrow();
    }
}

/// Add operation
impl std::ops::Add<&Value> for &Value {
    type Output = Value;

    fn add(self, rhs: &Value) -> Self::Output {
        let d = self.get_data() + rhs.get_data();
        let mut v = Value_::new(d, true);
        v.op = Op::Plus((*self).clone(), (*rhs).clone());
        Value(Rc::new(v))
    }
}

/// Mul
impl std::ops::Mul<&Value> for &Value {
    type Output = Value;

    fn mul(self, rhs: &Value) -> Self::Output {
        let d = self.get_data() * rhs.get_data();
        let mut v = Value_::new(d, true);
        v.op = Op::Mul((*self).clone(), (*rhs).clone());
        Value(Rc::new(v))
    }
}

#[cfg(test)]
mod tests {
    use graphviz_rust::cmd::CommandArg::Output;
    use graphviz_rust::dot_generator::*;
    use graphviz_rust::dot_structures::*;
    use graphviz_rust::{cmd::Format, exec, printer::PrinterContext};

    use crate::{calculate_grad, Op, Value};

    fn viz_computation_graph(value: &Value, graph: &mut Graph) {
        let value_node_id = value.id;
        let value_node = node!(
            value_node_id,
            vec![
                attr!("label", esc format!("{} | data={} grad={} {}",
                    value.id, value.get_data(), value.get_grad(),
                    { if !value.is_variable { "non-variable" } else { "" } }
                ))
            ]
        );
        graph.add_stmt(value_node.into());

        let viz_operand = |v1: &Value, g: &mut Graph| {
            let p_node_id = v1.id;
            let e = edge!(node_id!(p_node_id) => node_id!(value.id), vec![attr!("label", esc format!("{}", value.op))]);
            g.add_stmt(e.into());
            viz_computation_graph(v1, g);
        };

        match &value.op {
            Op::None => {}
            Op::Plus(v1, v2) | Op::Mul(v1, v2) => {
                viz_operand(v1, graph);
                viz_operand(v2, graph);
            }
            Op::Tanh(v1) => {
                viz_operand(v1, graph);
            }
        }
    }

    #[test]
    fn it_works() {
        let a = Value::new(-2.0);
        let b = Value::new(3.0);
        let d = &a * &b;
        let e = &a + &b;
        let f = &d * &e;
        let g = f.tanh();

        calculate_grad(&g);

        let mut graph = graph!(id!("computation"));
        viz_computation_graph(&g, &mut graph);
        let _graph_svg = exec(
            graph,
            &mut PrinterContext::default(),
            vec![Format::Png.into(), Output("./1.png".into())],
        )
        .unwrap();
    }
}
