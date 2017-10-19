### PID Project Report

#### PID Coefficients
PID coefficients ae used to reduce the cte(cross Track error) in a smooth way and these coefficients are calculated using cte values. In the project, the simulator provides cte values in the range of [-5,5] and the expected total_error(steer_value) is expected to be in the range of [-1,1]. As total_error is directly proportional to cte so the coefficient Kd is expected to be of the order of 0.2. Kd is the coefficient for the derivative term and this term depends upon the rate of cte change per time.Simulator is providing data every 0.1sec So to be of the same order, Kd is expected to be around 10 times Kp, so around 2.
Ki is the integral component, so it should be of the orde of 0.1 times Kp, so basically around 0.02.
So I started with the initial values of 0.2,0.02,2 and tried value combinations of values keeping throttle values of 0.3, 0.4 and 0.5. I realized that throttle also needs to be lower to make this work so I set throttle also equal to the absolute value of calculate steer_value with a cap at 0.3.
I finally kept the parameter values at Kp = 0.1, Ki = 0.01 and Kd = 2;

##### Significance of Coefficients
#### Ki Coefficient 
	- This controls the direct proportionality of cte error. So if the cte is large, Ki will bring it proportionally to the center(but it makes it overshoot.
	
#### Kd Coefficient 
	- This controls the direct proportionality of derivative of cte error with time. As it involves the cuent rate of change, it takes into account the future actions.

#### Ki Coefficient 
	- This controls the direct proportionality of integral of cte error over time. As it involves integral term, it sums over the past errors and helps in controlling even if the current input is weak.

