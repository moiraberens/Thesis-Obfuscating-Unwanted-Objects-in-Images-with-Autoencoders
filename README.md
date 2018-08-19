# Thesis-Obfuscating-Unwanted-Objects-in-Images-with-Autoencoders

In this repository you can find the code I have created  to run the experiments described in my thesis.
In my thesis I changed some names for clarity purpose in the code these names are not changed. Below you can find a list with all the names used in the code (left column) and the names they refer to in my thesis (right column).

private = unwanted
public = wanted
obf or obfuscator = autoencoder
att or attacker = discriminator
conv_deconv = autoencoder without skip-connections
unet = autoencoder with skip-connection

Additionally, there is a settings "extra block" in my code. 
If this settings is set equal to "skip", this mean an residual block.
If this settings is set equal to "true", this means an extra convolutional layer.
If this settings is set to "false", this means that no extra layer is added to the autoencoder.

