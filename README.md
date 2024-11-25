# Head Pose Detection

Head Pose Detection is the task of estimating the orientation of a person's head in 3D space relative to a camera or other reference point. 
This becomes relavant in fields such as virtual and augmented reality, human-computer interaction and securtiy systems.
Our project provides a simple way to find information about the head pose such as horizontal, vertical and inclined rotation values. 
This can be used to infer exactly which direction a subject has directed their head to.

## Working

We use the mediapipe facemesh to track varius feature points on a person's face. We then use specific points from this facemesh and using the relative distances of collinear points, we are able to estimate the aproximate angles or tilt of a person's head. 

## Demo

Image showing the tilt of a person's head.
![image](https://github.com/user-attachments/assets/cac3123f-052f-4e25-b66e-ee3def51765e)



## License

[MIT](https://choosealicense.com/licenses/mit/)
