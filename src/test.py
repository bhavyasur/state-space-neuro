import numpy as np

full_sess = np.array([[0.2, 0.3, 0.4, 0.6, 0.7],
                     [0.4, 0.6, 0.7, 0.8, 0.9],
                     [0.1, 0.2, 0.3, 0.4, 0.4],
                     [0.1, 0.1, 0.1, 0.1, 0.1]
                    ])
threshold = 0.5

for row in range(np.shape(full_sess)[0]):
    if full_sess[row, :].max(axis=-1) < threshold:
        full_sess[row, :] = 0
    else:
        full_sess[row, :] = full_sess[row, :]

full_sess_thresholded = full_sess[full_sess.any(axis=1)]
        
print(full_sess_thresholded)