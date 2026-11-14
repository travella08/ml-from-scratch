# linear regression from scratch

Linear regression trained with gradient descent in pure NumPy, verified against scikit-learn on synthetic data and California housing.

![The model's line after 0, 10, 50 and 500 epochs](assets/line_fitting.png)

## the math

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b \qquad L = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2$$

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{2}{n}\mathbf{X}^\top(\hat{\mathbf{y}} - \mathbf{y}) \qquad \frac{\partial L}{\partial b} = \frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)$$

## results

**Synthetic data** ($y = 5x + 10$ + noise): learned $w = 4.968$, $b = 10.037$, matching scikit-learn to 4 decimal places.

**California housing** (after standardizing the features):

| Model | Test MSE | Test $R^2$ |
|---|---|---|
| Mine | 0.555 | 0.577 |
| scikit-learn | 0.556 | 0.576 |

![Learning rate comparison](assets/lr_comparison.png)

## run it

```bash
pip install -r requirements.txt
```

Then open `experiments.ipynb`. The model is in `linreg.py`.

## what i learned

- How to derive gradients by hand and check them numerically
- Too large a learning rate makes training diverge, not just slow down
- Features on different scales need standardizing before gradient descent works
- Silent broadcasting bugs: always check array shapes
