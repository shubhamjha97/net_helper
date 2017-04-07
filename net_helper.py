import tensorflow as tf
import numpy as np

net_dict={'input':{'neurons':30, 'type':'ff', 'activation':'sigmoid', 'bias':'True'},
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
			elif layer_name=='output':
					self.output_dim=layer_attrs['neurons']
		self.input_layer=tf.placeholder(tf.float32, [None, self.input_dim])
		#self.output_layer=None
		self.y_=tf.placeholder(tf.float32, [None, self.output_dim])

		for i in range(0,(len(self.layer_dict)-1)):
			#print i
			if i==0: #check for first matrix
				#print self.layer_dict['input']['neurons'],'x', self.layer_dict['l'+str(i)]['neurons']
				self.W.append(tf.Variable(tf.zeros([self.layer_dict['input']['neurons'], self.layer_dict['l'+str(i)]['neurons']])))
				self.layers.append(tf.matmul(self.input_layer, self.W[i]))
			elif i==(len(self.layer_dict)-2): #check for last matrix
				#print self.layer_dict['l'+str(i-1)]['neurons'],'x', self.layer_dict['output']['neurons']
				self.W.append(tf.Variable(tf.zeros([self.layer_dict['l'+str(i-1)]['neurons'], self.layer_dict['output']['neurons']]))) 
				self.output_layer=(tf.matmul(self.layers[i-1], self.W[i]))
			else:
		 		#print self.layer_dict['l'+str(i-1)]['neurons'],'x', self.layer_dict['l'+str(i)]['neurons']
		 		self.W.append(tf.Variable(tf.zeros([self.layer_dict['l'+str(i-1)]['neurons'], self.layer_dict['l'+str(i)]['neurons']])))  
		 		self.layers.append(tf.matmul(self.layers[i-1], self.W[i]))
		 	
		if cost_fn=='CRSENT':
			self.cost=tf.reduce_mean(-tf.reduce_sum(self.y_*tf.log(self.output_layer+0.01), reduction_indices=[1]))

	def fit(self, X, y):
		self.train_step=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)
		pass
	def predict(self):
		pass

	def visualize(self):
		pass

if __name__ =="__main__":
	myNet=Net(net_dict)
	myNet.visualize()
	myNet.fit([[1,2], [5,3], [23,4], [23,456]], [0,0,1,0])