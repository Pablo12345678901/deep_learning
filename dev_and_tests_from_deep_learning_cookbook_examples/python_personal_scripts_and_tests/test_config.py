import tensorflow as tf
printf("CPU TEST")
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
printf("GPU TEST")
print(tf.config.list_physical_devices('GPU'))
