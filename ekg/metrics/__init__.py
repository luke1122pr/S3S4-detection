import tensorflow as tf

def cindex(y, risk):
    cs, st = tf.cast(y[:, 0:1], tf.float32), tf.cast(y[:, 1:2], tf.float32)

    risk_comparison_matrix = tf.subtract(tf.expand_dims(risk, -1), risk)

    risk_larger = tf.cast(risk_comparison_matrix > 0.0, tf.float32)
    risk_equal = tf.cast(tf.abs(risk_comparison_matrix) < 1e-3, tf.float32) * 0.5
    time_comparison = tf.cast(tf.subtract(tf.expand_dims(st, -1), st) < 0.0, tf.float32)
    ratio = tf.reduce_sum( (tf.reduce_sum(risk_larger * time_comparison, 1) + tf.reduce_sum(risk_equal * time_comparison, 1))*cs ) / tf.reduce_sum(tf.reduce_sum(time_comparison, 1) * cs)
    return ratio
