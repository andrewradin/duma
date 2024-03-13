import logging
logger = logging.getLogger(__name__)

def sigma_of_rank(scores, kt_idxs, n_kts, rank_sigmoid_scale, sigmoid_width=200, sigmoid_center=800):
    import tensorflow as tf
    scores = tf.reshape(scores, (-1,))
    ranks = kt_ranks(scores, kt_idxs, rank_sigmoid_scale)

    rank_scores = tf.sigmoid((sigmoid_center - ranks) / (0.5 * sigmoid_width))

    return tf.reduce_sum(rank_scores) / n_kts



def kt_ranks(scores, kt_idxs, sigmoid_scale):
    """
    Computes a 'soft' ranking of the specified KTs in the scoreset.

    Rank is computed for each KT by subtracting its score from all other scores
    and counting how many beat it.

    Rather than giving 1 or 0 for each, though, we use a sigmoid approximation.
    A score much higher then ours increases 'softrank' by 1, but if it's close
    it increases by less.  This allows our function to have a usable gradient.

    sigmoid_scale is the amount of 'softness' to the ranking.
    It is in score units - roughly an item's score needs to be sigmoid_scale
    higher than another for it to lose almost a full rank.  As it approaches 0
    it becomes a normal rank function.

    Exact ties are always 0.5 via sigmoid, aligning with our normal tie treatment.

    This returns the soft rank of the score at each kt_idx.
    """

    import tensorflow as tf

    # Pull out the scores of the KTs
    kt_scores = tf.gather(scores, kt_idxs)

    # Add some dimensions so we can use broadcasting to compute the delta for each KT from
    # all other scores.
    kt_scores = tf.expand_dims(kt_scores, 1)
    all_scores = tf.expand_dims(scores, 0)

    kt_deltas = all_scores - kt_scores

    #        Mol1   Mol2  ...
    # KT1    -0.9   0.3
    # KT2     0.1   0.2
    # ...


    # Apply a sigmoid as a smooth approximation to the heaviside/step function.
    # Each cell in each KT row becomes ~1 if that mol beat the KT, or ~0 if it did worse.
    kt_delta_sigmoids = tf.sigmoid(kt_deltas / (0.5 * sigmoid_scale))

    # The rank of each KT is the sum of its row
    # -0.5 to remove the tie with ourself that we get half a rank for
    # If we wanted ranks to start at 1 we could add 1 here, but for consistency
    # with EMInput's ranks, we don't.
    kt_ranks = tf.reduce_sum(kt_delta_sigmoids, axis=1) - 0.5

    return kt_ranks


def train(initial_weights, score_func, constrain_weights_func, iters=100, output_debug=False):
    # These have only been very lightly tuned via trial & error.
    GRAD_CLIP = 0.2
    LEARN_RATE = 0.03

    import tensorflow as tf
    import numpy as np

    opt = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)

    weights = tf.Variable(np.reshape(initial_weights, (-1, 1)))
    init_score = score_func(weights, 1.0)
    init_score_str = [('%.4f' % x) for x in init_score]

    @tf.function
    def one_step(weights, iter_num):
        with tf.GradientTape() as g:
            g.watch(weights)
            out_score, unpen_score = score_func(weights, iter_num/iters)
        grads = g.gradient(out_score, weights)
        # Apply some gradient clipping, we see sporadic extreme gradients.
        grads = tf.clip_by_value(grads, -GRAD_CLIP, GRAD_CLIP)
        opt.apply_gradients([(-grads, weights)])
        constrain_weights_func(weights)
        return out_score, unpen_score, grads
    
    grad_accum = []
    weight_accum = []

    weight_accum.append(weights.numpy().reshape(-1))
    for i in range(iters):
        out_score, unpen_score, grads = one_step(weights, iter_num=tf.constant(i),
                )
        
        if output_debug:
            grad_accum.append(grads.numpy().reshape(-1))
            weight_accum.append(weights.numpy().reshape(-1))
    
    logger.info("Score from %s to %.4f, %.4f over %d iters", init_score_str, out_score, unpen_score, iters)

    import numpy as np
    if init_score[0] > out_score or np.isnan(out_score):
        logger.info(f"Reverting to initial score {init_score[0]} vs {out_score}")
        return initial_weights, grad_accum, weight_accum

    return np.reshape(weights.numpy(), (-1,)), grad_accum, weight_accum

