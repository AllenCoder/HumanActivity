import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

print(node1, node2)  # 只打印节点信息

sess = tf.Session()
# print(sess.run([node1,node2])) # 运行节点
sess.run(tf.global_variables_initializer())

print(sess)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 占位量
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # 与调用add方法类似
print(sess.run(adder_node, {a: 3, b: 4.5}))

print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# 变量

W = tf.Variable([.3], tf.float32)
b = tf.Variable([.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
sess.run(tf.global_variables_initializer())
print("linear_model: ", sess.run(linear_model, {x: [1, 2, 3, 4]}))
