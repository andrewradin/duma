
def test_constraints():
    from scripts.wzs import WeightConstraint
    import tensorflow as tf
    
    constraint = WeightConstraint([1, 3], 10)

    weights = tf.Variable([1.3, 5.0, 3.2, 15.0, 7.4], dtype=tf.float64)

    constraint.tf_apply_to(weights, 5)
    print("Weight are now ", weights)
