# *A-MAML*: Meta-Learning with Adjoint Methods

by [Shibo Li](https://imshibo.com), Wang Zheng, [Akil Narayan](http://www.sci.utah.edu/~akil/), [Mike Kirby](https://www.cs.utah.edu/~kirby/) and [Shandian Zhe](https://www.cs.utah.edu/~zhe/)

<p align="center">
    <br>
    <img src="images/amaml-idea-trim.png" width="800" />
    <br>
<p>

<h4 align="center">
    <p>
        <a href="https://arxiv.org/abs/2110.08432">Paper</a> |
        <a href="https://github.com/shib0li/Infinite-Fidelity-Coregionalization/blob/main/images/amaml-slides.pdf">Slides</a> 
<!--         <a href="https://github.com/shib0li/Infinite-Fidelity-Coregionalization/blob/main/images/poster.pdf">Poster</a>  -->
    <p>
</h4>


Model Agnostic Meta Learning (MAML) is widely used to find a good initialization for a family of tasks. Despite its success, a critical challenge in MAML is to calculate the gradient w.r.t. the initialization of a long training trajectory for the sampled tasks, because the computation graph can rapidly explode and the computational cost is very expensive. 

To address this problem, we propose Adjoint MAML (A-MAML). We view gradient descent in the inner optimization as the evolution of an Ordinary Differential Equation (ODE). To efficiently compute the gradient of the validation loss w.r.t. the initialization, we use the adjoint method to construct a companion, backward ODE. To obtain the gradient w.r.t. the initialization, we only need to run the standard ODE solver twice --- one is forward in time that evolves a long trajectory of gradient flow for the sampled task; the other is backward and solves the adjoint ODE. We need not create or expand any intermediate computational graphs, adopt aggressive approximations, or impose proximal regularizers in the training loss.  Our approach is cheap, accurate, and adaptable to different trajectory lengths. We demonstrate  the advantage of our approach in both synthetic and real-world meta-learning tasks. 


# System Requirements

We highly recommend to use Docker to run our code. We have attached the docker build file `env.Dockerfile`. Or feel free to install the packages with pip/conda that could be found in the docker file.


# Run

To run 

```
bash experiment.sh $DOMAIN $k_SHOT $k_QUERY $INNER_STEPS $DEIVCE 
```

* `$DOMAIN` name of problem
* `$k_SHOT` number of shots
* `$k_QUERY` number of support
* `$INNER_STEPS` SGD steps in the inner loop
* `$DEVICE` where to run, for example ***cuda:0*** or ***cpu***

For example

```
bash experiment.sh Alpine2D 100 100 200 cuda:0 
```
will get the results of meta regression on family of Alpine functions with 100 shots and 100 query, each meta step will run 200 SGD steps.


# License

Adjoint MAML is released under the MIT License, please refer the LICENSE for details

# Getting Involved
Feel free to submit Github issues or pull requests. Welcome to contribute to our project!

To contact us, never hestitate to send an email to `shibo@cs.utah.edu` or `shiboli.cs@gmail.com` 
<br></br>


# Citation
Please cite our paper if you find it helpful :)

```

@article{li2021meta,
  title={Meta-Learning with Adjoint Methods},
  author={Li, Shibo and Wang, Zheng and Narayan, Akil and Kirby, Robert and Zhe, Shandian},
  journal={arXiv preprint arXiv:2110.08432},
  year={2021}
}

```
<br></br>
