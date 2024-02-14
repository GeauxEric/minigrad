use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::Formatter;
use std::iter::zip;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rand::Rng;

#[derive(Debug, Clone)]
enum Op {
    None,
    Plus(Value, Value),
    Mul(Value, Value),
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
    let rev_tp_order = reverse_topological_order(root.clone());
    for v in &rev_tp_order {
        match &v.op {
            Op::None => {}
            Op::Plus(v1, v2) => {
                // v = v1 + v2
                // d(v) / d(v1) = 1
                // d(L) / d(v1) = d(L) / d(v) * d(v) / d(v1) = parent_grad * 1.0
                *v1.grad.borrow_mut() += v.get_grad();
                *v2.grad.borrow_mut() += v.get_grad();
            }
            Op::Mul(v1, v2) => {
                // v = v1 * v2
                // d(v) / d(v1) = v2
                // d(L) / d(v1) = d(L) / d(v) * d(v) / d(v1) = parent_grad * v2
                *v1.grad.borrow_mut() += v.get_grad() * v2.get_data();
                *v2.grad.borrow_mut() += v.get_grad() * v1.get_data();
            }
        }
    }
}

fn get_id() -> usize {
    static COUNTER: AtomicUsize = AtomicUsize::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

impl Value {
    pub fn new(data: f32) -> Self {
        Value(Rc::new(Value_::new(data)))
    }
}

impl Value_ {
    pub fn new(data: f32) -> Self {
        let id = get_id();
        Value_ {
            data: RefCell::new(data),
            id,
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
        let mut v = Value_::new(d);
        v.op = Op::Plus((*self).clone(), (*rhs).clone());
        Value(Rc::new(v))
    }
}

/// Mul
impl std::ops::Mul<&Value> for &Value {
    type Output = Value;

    fn mul(self, rhs: &Value) -> Self::Output {
        let d = self.get_data() * rhs.get_data();
        let mut v = Value_::new(d);
        v.op = Op::Mul((*self).clone(), (*rhs).clone());
        Value(Rc::new(v))
    }
}

fn reverse_topological_order(value: Value) -> Vec<Value> {
    let mut order = topological_order(value);
    order.reverse();
    order
}

fn topological_order(value: Value) -> Vec<Value> {
    let mut order = vec![];
    let mut visited = HashSet::new();
    fn build_topo(value: Value, visited: &mut HashSet<usize>, order: &mut Vec<Value>) {
        if !visited.contains(&value.id) {
            visited.insert(value.id);
            match &value.op {
                Op::None => {}
                Op::Plus(v1, v2) => {
                    build_topo(v1.clone(), visited, order);
                    build_topo(v2.clone(), visited, order);
                }
                Op::Mul(v1, v2) => {
                    build_topo(v1.clone(), visited, order);
                    build_topo(v2.clone(), visited, order);
                }
            }
            order.push(value)
        }
    }
    build_topo(value, &mut visited, &mut order);
    order
}

#[derive(Debug)]
struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::thread_rng();
        let range = -1.0f32..=1.0;
        let random_numbers: Vec<f32> = (0..nin + 1)
            .map(|_| {
                let f = rng.gen_range(range.clone());
                f
            })
            .collect();
        Neuron {
            weights: random_numbers[..nin]
                .iter()
                .map(|f| Value::new(*f))
                .collect(),
            bias: Value::new(random_numbers[nin]),
        }
    }

    // TODO: add non-linear activation
    pub fn apply(&self, x: &[Value]) -> Value {
        assert_eq!(
            x.len(),
            self.weights.len(),
            "length of input vector not equal to length of weights"
        );
        let mut s = Value::new(0.0);
        for (xi, wi) in zip(x, &self.weights) {
            s = &s + &(xi * wi);
        }
        s = &s + &self.bias;
        s
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(nin)).collect(),
        }
    }

    pub fn apply(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.apply(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use graphviz_rust::cmd::CommandArg::Output;
    use graphviz_rust::dot_generator::*;
    use graphviz_rust::dot_structures::*;
    use graphviz_rust::{cmd::Format, exec, printer::PrinterContext};

    use crate::{calculate_grad, reverse_topological_order, topological_order, Layer, Op, Value};

    fn viz_computation_graph(value: &Value, graph: &mut Graph) {
        let reverse_tp_order = reverse_topological_order(value.clone());

        for value in &reverse_tp_order {
            let value_node_id = value.id;
            let value_node = node!(
                value_node_id,
                vec![
                    attr!("label", esc format!("{} | data={} grad={} op={}", value.id, value.data.borrow(), value.grad.borrow(), value.op))
                ]
            );
            graph.add_stmt(value_node.into());
            let mut add_edge = |v: &Value| {
                let p_node_id = v.id;
                let e = edge!(node_id!(p_node_id) => node_id!(value_node_id));
                graph.add_stmt(e.into());
            };
            match &value.op {
                Op::None => {}
                Op::Plus(v1, v2) => {
                    add_edge(v1);
                    add_edge(v2);
                }
                Op::Mul(v1, v2) => {
                    add_edge(v1);
                    add_edge(v2);
                }
            }
        }
    }

    #[test]
    fn mlp() {
        let n1 = Layer::new(4, 2);
        let x = vec![
            Value::new(1.0),
            Value::new(2.0),
            Value::new(3.0),
            Value::new(4.0),
        ];
        let r = n1.apply(&x);
        let mut graph = graph!(id!("neuron"));
        for v in &r {
            viz_computation_graph(v, &mut graph);
        }
        let _graph_svg = exec(
            graph,
            &mut PrinterContext::default(),
            vec![Format::Png.into(), Output("./neuron.png".into())],
        )
        .unwrap();
    }

    #[test]
    fn topo_order() {
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let c = &a + &b;
        let d = Value::new(3.0);
        let e = &c + &d;

        let order = topological_order(e);
        let ids: Vec<_> = order.into_iter().map(|v| v.id).collect();
        println!("{:?}", ids);
    }

    #[test]
    fn it_works() {
        let a = Value::new(-2.0);
        let one = Value::new(1.0);
        let c = &a * &one;
        let b = &(&a * &one) + &c;
        let f = &b * &c;

        calculate_grad(&f);

        let mut graph = graph!(id!("computation"));
        viz_computation_graph(&f, &mut graph);
        let _graph_svg = exec(
            graph,
            &mut PrinterContext::default(),
            vec![Format::Png.into(), Output("./1.png".into())],
        )
        .unwrap();
    }
}
