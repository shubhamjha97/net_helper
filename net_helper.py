import tensorflow as tf
import numpy as np
import webbrowser

#Use batch size, batch sampling!!!!!!!!!!!!!!!!!!!!
net_dict={'input':{'neurons':2, 'type':'ff', 'activation':'sigmoid', 'bias':'True'},
			'l0':{'neurons':100, 'type':'ff', 'activation':'sigmoid', 'bias':'True'},
			'l1':{'neurons':80, 'type':'ff', 'activation':'sigmoid', 'bias':'True'},
			'l2':{'neurons':50, 'type':'ff', 'activation':'sigmoid', 'bias':'True'},
			'output':{'neurons':10, 'type':'ff', 'activation':'softmax', 'bias':'False'}
			}
class Net:

	'''net dict- input layer
	CRSENT= cross-entropy
	ff- feedforward'''

	def __init__(self, net_struct, cost_fn='CRSENT', learning_rate=0.01, epochs=100, batch_size=1000):
		self.layer_dict=net_struct
		self.W=[]
		self.layers=[]
		self.learning_rate=learning_rate
		self.epochs=epochs
		self.batch_size=batch_size
		for layer_name, layer_attrs in self.layer_dict.iteritems():
			if layer_name=='input':
					self.input_dim=layer_attrs['neurons']
					print 'input_dim=', self.input_dim
			elif layer_name=='output':
					self.output_dim=layer_attrs['neurons']
					print 'output_dim=', self.output_dim
		self.input_layer=tf.placeholder(tf.float32, [None, self.input_dim])
		self.y_=tf.placeholder(tf.float32, [None, self.output_dim])

		for i in range(0,(len(self.layer_dict)-1)):
			#print i
			if i==0: #check for first matrix
				with tf.name_scope('input_layer'):
					if len(self.layer_dict)==2:
						print self.layer_dict['input']['neurons'],'x', self.layer_dict['output']['neurons']
						self.W.append(tf.Variable(tf.zeros([self.layer_dict['input']['neurons'], self.layer_dict['output']['neurons']])))
						self.output_layer=tf.nn.sigmoid(tf.matmul(self.input_layer, self.W[i]))
						self.layers.append(tf.nn.sigmoid(tf.matmul(self.input_layer, self.output_layer)))

					else:
						print self.layer_dict['input']['neurons'],'x', self.layer_dict['l'+str(i)]['neurons']
						self.W.append(tf.Variable(tf.zeros([self.layer_dict['input']['neurons'], self.layer_dict['l'+str(i)]['neurons']])))
						self.layers.append(tf.nn.sigmoid(tf.matmul(self.input_layer, self.W[i])))
					
			elif i==(len(self.layer_dict)-2): #check for last matrix
				with tf.name_scope('output_layer'):
					if len(self.layer_dict)==2:
						print self.layer_dict['l'+str(i-1)]['neurons'],'x', self.layer_dict['output']['neurons']
				 		self.W.append(tf.Variable(tf.zeros([self.layer_dict['l'+str(i-1)]['neurons'], self.layer_dict['output']['neurons']]))) 
				 		self.output_layer=tf.nn.sigmoid(tf.matmul(self.input_layer, self.W[i]))
				 	else:
				 		self.W.append(tf.Variable(tf.zeros([self.layer_dict['input']['neurons'], self.layer_dict['output']['neurons']]))) 
				 		self.output_layer=tf.nn.sigmoid(tf.matmul(self.layers[i-1], self.W[i]))
			else:
			 	with tf.name_scope('layer_'+str(i-1)):
		  			#print self.layer_dict['l'+str(i-1)]['neurons'],'x', self.layer_dict['l'+str(i)]['neurons']
		  			self.W.append(tf.Variable(tf.zeros([self.layer_dict['l'+str(i-1)]['neurons'], self.layer_dict['l'+str(i)]['neurons']])))  
		  			self.layers.append(tf.nn.sigmoid(tf.matmul(self.layers[i-1], self.W[i])))
		with tf.name_scope('cost'):
			if cost_fn=='CRSENT':
				self.cost=tf.reduce_mean(-tf.reduce_sum(self.y_*tf.log(self.output_layer+0.01), reduction_indices=[1]))
			self.train_step=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

		# with tf.Session() as sess:
		# 	sess.run(tf.global_variables_initializer())
		# 	print 'Variables initialized' 	
		
	def fit(self, X, y):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print 'Variables initialized....'
			for epoch in range(self.epochs):
				cost, _= sess.run([self.cost, self.train_step], feed_dict={self.input_layer:X, self.y_:y})
			print 'Training complete....'

	def predict(self, X_test):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print 'Variables initialized....'
			predictions=sess.run(self.output_layer, feed_dict={self.input_layer:X_test})
		return predictions


	def visualize(self):
		with tf.Session() as sess:
			writer=tf.summary.FileWriter("./logs/nn_logs", graph=tf.get_default_graph())
			merged=tf.summary.merge_all()
			webbrowser.open('http://127.0.1.1:6006', new=2)
		

if __name__ =="__main__":
	myNet=Net(net_dict)
	myNet.visualize()