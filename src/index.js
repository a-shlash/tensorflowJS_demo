import * as tf from "@tensorflow/tfjs";
// -----------------//
// Create Tensor    //
// -----------------//

// 1-D tensor
const t = tf.tensor([1, 2, 3, 4]);

console.log(t.rank);
console.log(t.shape);
// print tensor
console.log(t);
console.log(t.dataSync());

// print the data in tensor
t.print();

// 2-D tensor
const t2d = tf.tensor2d([
  [1, 2, 3],
  [4, 5, 6]
]);
console.log(t2d.rank);
console.log(t2d.shape);
// print tensor
console.log(t2d);

// print the data in tensor
t2d.print();
// -----------------//
// Functions Utiles //
// -----------------//
//1-zeros: Tensor with a metrix with dimentions 3 rows  & 4 columns and all values = zero
tf.zeros([3, 4]).print();

//2-ones: Tensor with a metrix with dimentions 3 rows  & 4 columns and all values = one
tf.ones([3, 4]).print();

// 3 - reshape
const ts = tf.tensor([1, 2, 3, 4, 5, 6]);
ts.print();
ts.reshape([3, 2]).print();

// Add
//const a = tf.tensor([1, 2, 3]);
//const b = tf.tensor([4, 5, 6]);
//
//a.add(b).print()

const c = tf.tensor2d([
  [1, 2, 3],
  [4, 5, 6]
]);
c.print();
c.square().print(); // or tf.dot(a, b)

//Produit scalaire
const d1 = tf.tensor2d([
  [1, 2],
  [1, 2]
]);

const d2 = tf.tensor2d([
  [3, 4],
  [3, 4]
]);
d1.print();
d2.print();

d1.dot(d2).print();

// Multiplicaiton 1
d1.mul(d2).print();

// Multiplicaiton 2
d1.matMul(d2).print();

// Multiplicaiton 3
const mat1 = tf.tensor2d([
  [1, 2, 3],
  [4, 5, 6]
]);
const mat2 = tf.tensor2d([
  [7, 8, 9],
  [10, 11, 12]
]);
mat1.print();
mat2.print();
//mat1.matMul(mat2).print();

/// Transpose
const transMat2 = mat2.transpose();
mat1.print();
transMat2.print();
mat1.matMul(transMat2).print();

//Tensor Disposal

const x = tf.tensor([1, 2, 3]);
const x2 = x.square().square();
x.print();
x2.print();
x.dispose();
//x.print();
