# Code for Thesis "Obfuscating Unwanted Objects in Images with Autoencoders"
In this repository, you can find the code I have created to run the experiments described in my thesis.

###### Names that differ between code and thesis
In my thesis, I changed some names for clarity purpose in the code these names are not changed. Below you can find a list of all the names used in the code (left) and the names they refer to in my thesis (right).

 - private = unwanted
 - public = wanted
 - obf or obfuscator = autoencoder
 - att or attacker = discriminator
 - conv_deconv = autoencoder without skip-connections
 - unet = autoencoder with skip-connection

Additionally, there is a settings "extra block" in my code. 
If this setting is set equal to "skip", this means a residual block is added after every layer of the autoencoder.
If this setting is set equal to "true", this means an extra convolutional layer is added after every layer of the autoencoder.
If this setting is set to "false", this means that no extra layer is added to the autoencoder.

###### Data

