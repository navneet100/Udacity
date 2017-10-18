I started to explore first with the loadbottleneck data but I was not making much progress. So I then took nvidia model as
base and put additional features of dropout. 

I used model with 5 Convnet layers, 3 FC layers and one output layer.

4 Convnet Layers :
	1) 5 x 5 filter with strides of 2 and output depth of 24
	2) 5 x 5 filter with strides of 2 and output depth of 36
	3) 5 x 5 filter with strides of 2 and output depth of 48
	4) 3 x 3 filter with strides of 2 and output depth of 64
		Dropout(0.5)
	5) 3 x 3 filter with strides of 2 and output depth of 64	
	
	Dropout(0.5) was added after 4th Convnet Layer

3 FC Layers
	1) 100 neurons
	2) 50 neurons
		Dropout(0.5)
	3) 10 neurons
	
	Dropout(0.5) was added after 3nd FC Layer
		

Outlayer Layer 		
	1) 1 Neuron(Class)

Activation used was Relu.It was added after each layer For this model and data, I tried with elu also, but it was not working	
	
Dropout was not mentioned in Nvidia paper but I tried adding this.
First I added dropuout after every layer but it was not working, so then I removed most of the dropouts and kept only 2. 
One after 4th Convnet layer and one after 2nd Fully Connected layer.

Optimizer - I used Adam optimizer with learning rate of 0.0001 

Keras Data generator was used and batch size was 64.
Samples_per_epoch - 20096 ( multiple of batch size = 64)


I took Udacity provided data and added some of my own data.
I added some data as the model was not working with the Udacity provided data.
I generated my own data and added this to the Udacity provided dataset. After number of
iterations of data generation, I have been able to run this successfully for couple laps.
I have learned that this project requires not just tuning of parameters but proper training dataset also otherwise 
the model learns different behaviour.

I tested it on Track 1 only. I have been able to run the car a couple of laps.

After combining new data with existing Udacity data, it was split into training and validation
dataset using pandas sampling method.

PreProcessing of images involved 
	1) adding shift of 0.17 to left and right images.
	2) adding brightness
	3) adding flip randomly
	4) cropped images from top and bottom to exclude horizon and hood
	5) finally resized image to 16 x 32. I saw that this size gave almost same results as larger sizes (66, 200), (100,200 ) but as training was faster, so i used 16 * 32 and it was also explored by someone in forum.
	
For preprocessing, I used some of the methods suggested by people on forums and made changes to these for refinement.




