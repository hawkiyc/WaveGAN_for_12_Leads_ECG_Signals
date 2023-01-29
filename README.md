# WaveGAN for 12 Leads ECG Signals

12 Leads ECG is essential for identifying arrhythmia and other cardio malfunctions. However, it is always tricky to collect large amount of ECG data. With Deep Learning, generated ECG data can solve the problems of lacking data. In all candidates of generated model, GAN shows high performance in CV area. However, due to the nature of time series data, it is harder to generate long sequence than big picture before WaveGAN architecture. Here by, I bulid a WaveGAN model for create artificial 12 Leads ECG data. 

Here are results for this WaveGAN model
![12](https://user-images.githubusercontent.com/76748651/215320318-817564a6-a8a1-4108-81da-65de5e5ec138.png)
![14](https://user-images.githubusercontent.com/76748651/215320319-3bbd9b65-2ebd-48f2-a08b-8dcf8f421a0f.png)
![18](https://user-images.githubusercontent.com/76748651/215320321-49430092-bedc-4f13-89d6-647dd479c81b.png)
![20](https://user-images.githubusercontent.com/76748651/215320323-bf479832-b3be-4c25-bd80-fb6e64063542.png)
![28](https://user-images.githubusercontent.com/76748651/215320325-5407eca0-04e9-464d-bd4a-b303e10d73e9.png)

# Dataset Description 

I built this model with Code-15% dataset. You can find the description and original data from following link.
https://zenodo.org/record/4916206

You can also download cleaned data for this particular project via following Link:
https://1drv.ms/u/s!ArQCikHAsFj6oPJoQY7h9rIggjoaug?e=YjNprR

# Reference:

1. https://github.com/mostafaelaraby/wavegan-pytorch

2. Adversarial Audio Synthesis: https://arxiv.org/abs/1802.04208

3. A Generative Adversarial Approach To ECG Synthesis And Denoising: https://www.researchgate.net/publication/344159398_A_Generative_Adversarial_Approach_To_ECG_Synthesis_And_Denoising
