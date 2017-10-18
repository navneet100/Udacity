Methodology:

Training Data:
1) Initially used Udacity Training data without any augmentation and new data
2) Split training data into training(90%) and validation(10%) using scikit train_test_split
3) As training data has steering angle for only center camera, so steering angle for left and right
	cameras was generated on the fly by adding\subtracting shift(0.17)from the center image steering angle.
4) Keras model generator was used to take samples from the training data.128 images were used in one batch.
5) Took 20096 samples per epoch, total 5 epochs were processed.
6) Started with Nvidia model, changed layer depths, added additional layers, added dropouts after each layer.
	As there was no change in testing car driving performance, I took that it is not the model but something 
	else that is not working. So I moved back to the base nvidia model and worked on images augmentation.
7) Images were augmented as suggested by Vivek Yadav on his post. 
	Ref: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.uo4shtokb
8) This resulted in some progress but even then performance did not improve much.
9) Finally I decided to generate some new data. 
	a) Initially I just drove straight with not much deviations from center ( not much improvement)
	b) Then I took the car to multiple off road locations, Started recording while taking car back to the
		main road (instead of progress, it resulted in poor performance and car started going offtrack near 
		the points where I had kept car offtrack in training mode)
	c) Then I decided to move car slightly towards the boundary and as it was approaching the boundary(yellow
    	line), i started recording bringing car back to the middle of the road. I did this a number of times at different locations in the lap. This resulted in big improvement.
	d) I noted down the locations where the car was still going offtrack, I generated more data as mentioned in step c and it finally gave a big improvement.
10) Final driving_log file had 25118 entries ( approx 3 times the original Udacity data)

Model Exploration:
1) Started with LoadBottleneck data but got stuck and was not making any progress
2) Read nvidia paper and started with this model
	ref:http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
3) Changed depths, added layers (one more convnet and one more FC layer) - still no dropouts as these were not mentioned in paper.
	Not much change in improvement using following models
	a) modified nvidia - Tried just first 2 convnet layers and removed the 50 neurons FC layer
	b) convnet layers with depths ( 24, 48,96,192, 64) followed by FC layers with neurons(256,64,16,1), dropout of 0.5 after each layer
	c)convnet layer with maxpool ( depths - 32,64,128, 256) followed by FC layers with neurons(256,64,16,1), dropout of 0.5 after some layers and 0.33 after some layer
4) As with original data, there was no progress so switched back to the original nvidia model.
5) Augmented the images and added more training data and it improved performance.
6) To further improve the performance, I added dropouts after every layer, but it did  not work. So I added dropouts
	after 4th convnet layer and after second FC layer. This worked and performance was satisfactory.
7) Final model used is as mentioned below:
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

8) Activation used was Relu.It was added after each layer For this model and data, I tried with elu also, but it was not working ( originally with no 			
   additional data, I will try elu again with newly generated data)
		
9) Optimizer - I used Adam optimizer with learning rate of 0.0001. I tried 0.001 also but it was before the additional training data

Images Augmentation:
1) Initially I just used steering angle shift for left and right camera images
2) Then as suggested by Vivek yadav:
	a) Added brigtness to images
	b) Added shadow to the images
	c) Flipped the images
	d) translated images
	e) Top 60 and bottom 20 pixels were cropped
	f) finally image was resized to 16x32
3) This processing was done on the fly and images were generated using Keras generator. 128 images 
	were used in one batch.	Took 20096 samples per epoch, total 5 epochs were processed.
4) nvidia paper suggested 66x200 size image. I started with 66x200 size but it was taking too much 
	processing time, so reduced the image size to 16x32 after all processing. Each epoch was taking 
	around 5 to 10 minutes. 

drive.py change
I made changes in the drive.py file so that 
1) car runs at slow speed( throttle is set to 0.1)
2) top 60 and bottom 20 pixels are cropped and image is resized to 16x32
3) model is compiled with Adam optimizer with lr = 0.0001



Future Plan :
1) Generate more data
2) Try again different model configs ( other than nvidia), add , remove convnet, FC layers.As I think for me, these different models did not work because i did not use proper training data, I will run again my earlier models on this training data and see if these models work now
3) Augment images so that testing works on track 2 also.
4) Again try transfer learning and loadbottleneck data approaching



