\section{Optimal Flow Matching (OFM)}\label{sec:OFM}
\vspace{-2mm}
In this section, we provide the design of our novel Optimal Flow Matching algorithm \eqref{alg:OFM} that fixes main problems of Rectified Flow and OT-CFM approaches described above. In theory, it obtains exactly \textbf{straight trajectories} and recovers the unbiased optimal transport map for the quadratic cost \textbf{just in one FM iteration} with \textbf{any} initial transport plan. Moreover, during inference, our OFM does not require solving ODE to transport points.

We discuss the theory behind our approach  ($\S$\ref{sec:deriving loss}), its practical implementation aspects  ($\S$\ref{sec:practice}) and the relation to prior works ($\S$\ref{sec:prior}). All \underline{our proofs} are located in Appendix~\ref{sec:proofs}.

\vspace{-2mm}
\subsection{Theory: Deriving the Optimization Loss}\label{sec:deriving loss}
\vspace{-2mm}

We want to design a method of moving distribution $p_0$ to $p_1$ via exactly straight trajectories. Namely, we aim to obtain straight paths from the solution of the dynamic OT \eqref{eq:dynamic ot}.
%with the quadratic cost function $c(x_0, x_1) = \frac{\|x_0 - x_1\|^2}{2}$.
Moreover, we want to limit ourselves to just one minimization iteration. Hence, we propose our novel Optimal Flow Matching (OFM) procedure satisfying the above-mentioned conditions. The main idea of our OFM is to minimize the Flow Matching loss \eqref{eq:rectflow min} not over all possible vector fields $u$, but only over specific \textit{optimal} ones, which yield straight paths by construction and include the desired  dynamic OT field $u^*$.



\textbf{Optimal vector fields.} We say that a vector field $u^\Psi$ is optimal if it generates linear trajectories  $\{\{z_t\}_{t\in [0,1]}\}$ such that there exist a convex function $\Psi: \R^D \to \R$, which for any path $\{z_t\}_{t\in[0,1]}$ pushes the initial point $z_0$ to the final one as $z_1 = \nabla \Psi(z_0)$, i.e.,
\begin{eqnarray}
    z_t &=& (1-t) z_0  + t \nabla \Psi(z_0), \quad t \in [0,1].\notag
\end{eqnarray}
The function $\Psi$ defines the ODE
\begin{eqnarray}
    dz_t &=& (\nabla \Psi(z_0) - z_0)dt, \quad  z_t|_{t=0} = z_0.\label{eq:convex traj ODE} 
\end{eqnarray}
Equation \eqref{eq:convex traj ODE} does not provide a closed formula for 
 $u^\Psi$ as it depends on $z_0$. The explicit formula is constructed as follows: for a time $t \in [0,1]$ and point $x_t$, we can find a trajectory $\{z_t\}_{t \in [0,1]}$ s.t.
 \begin{equation}\label{eq:init point eq}
     x_t = z_t = (1-t) z_0  + t \nabla \Psi(z_0)
 \end{equation}

\begin{wrapfigure}{r}{0.43\textwidth}
\vspace{-5mm}
\hspace*{2mm}\begin{subfigure}[b]{0.97\linewidth}
\centering
\includegraphics[width=1.0\linewidth]{figs/optimal_fileds.pdf}
\vspace{0.51mm}
\end{subfigure}
\vspace{-7mm}\caption{\centering \small An Optimal Vector Field: a vector field $u^\Psi$ with straight paths is parametrized by a gradient of a convex function $\Psi$.}
\label{fig:optimal fields}
\vspace{-5mm}
\end{wrapfigure}

 and recover the initial point $z_0$. We postpone the solution of this problem to $\S$\ref{sec:practice}. For now, we  define the inverse of flow map \eqref{eq:push-forward operator} as $(\phi^\Psi_t)^{-1} (x_t) := z_0$ and the vector field $u_t^\Psi(x_t) := \nabla \Psi(z_0) - z_0 = \nabla \Psi((\phi^\Psi_t)^{-1} (x_t)) - (\phi^\Psi_t)^{-1} (x_t) $, which generates ODE \eqref{eq:convex traj ODE}, i.e., $dz_t = u^\Psi_t(z_t)dt.$ The concept of optimal vector fields is depicted on Figure \ref{fig:optimal fields}. %\petr{Fix figure!}
We highlight that the solution of dynamic OT lies in the class of optimal vector fields, since it generates linear trajectories \eqref{eq:dynamic ot linear traj} with the Brenier potential $\Psi^*$ \eqref{eq:opt map via grad}.


 \textbf{Training objective.}  
Our Optimal Flow Matching (OFM) approach is as follows: we restrict the optimization domain of FM  \eqref{eq:rectflow min} with fixed plan $\pi$ only to the optimal vector fields. We put the formula for the vector field $u_{\Psi}$ into FM loss from \eqref{eq:rectflow min} and define our Optimal Flow Matching loss:
\begin{eqnarray}\label{eq:OFM loss} 
    \mathcal{L}_{OFM}^\pi(\Psi) \!\!&:=& \!\!\mathcal{L}_{FM}^\pi(u^\Psi)\!\!=\!\!  \int\limits_0^1\!\!\left\{\int\limits_{\R^D \times \R^D}  \!\!  \|   u^{\Psi}_t(x_t) - (x_1 - x_0)\|^2 \pi(x_0, x_1) dx_0 dx_1\!\!\right\}\!\!dt, \\
     x_t &=& (1-t)  x_0 + t x_1. \notag
\end{eqnarray}

Our Theorem \ref{thm:loss equiv} states that OFM solves the dynamic OT via single FM minimization for any initial $\pi$.
\begin{theorem}[OFM and OT connection]\label{thm:loss equiv}
    Consider two distributions  $p_0, p_1 \in \mathcal{P}_{ac, 2}(\R^D)$ and \textbf{any} transport  plan $\pi \in \Pi(p_0, p_1)$ between them. Then, the dual Optimal Transport loss $\mathcal{L}_{OT}$ \eqref{eq:dual formulation} and Optimal Flow Matching loss $\mathcal{L}^\pi_{OFM} $ \eqref{eq:OFM loss}  have \textbf{the same minimizers}, i.e.,
    \begin{equation}
        \argmin\limits_{\text{convex }\Psi } \mathcal{L}^\pi_{OFM}(\Psi) = \argmin\limits_{\text{convex }\Psi } \mathcal{L}_{OT}(\Psi).\notag
    \end{equation}
\end{theorem}

\vspace{-2mm}
\subsection{Practical implementation aspects}\label{sec:practice}
\vspace{-2mm}
In this subsection, we explain the details of optimization of  our Optimal Flow Matching loss \eqref{eq:OFM loss}. 


\textbf{Parametrization of $\Psi$.} In practice, we parametrize the class of convex functions with Input Convex Neural Networks (ICNNs) \cite{amos2017input} $\Psi_\theta$  and parameters $\theta$. These
are scalar-valued neural networks built in such a way that the network is convex in its input. They consist of fully-connected or convolution blocks, some weights of which are set to be non-negative in order to keep the convexity. In addition, activation functions are considered to be only non-decreasing and convex in each input coordinate. %It is possible because of the next property of composition for convex functions: if function $f: \R^D \to R$ is convex non-decreasing and $g:\R^D \to \R^D$ is convex in each output coordinate, then $f \circ g$ is convex as well. 
These networks are able to support most of the popular training techniques (e.g., gradient descent optimization, dropout, skip connection, etc.). In Appendix \ref{sec:app exp}, we discuss the used \underline{architectures}.



\textbf{OFM loss calculation.} 

\begin{proposition}[Explicit Loss Gradient Formula]\label{rmk:loss in practice}
 The gradient of $\mathcal{L}^\pi_{OFM}$ can be calculated as 
 \begin{eqnarray}
    z_0 &=& \textsc{NO-GRAD}\left\{(\phi^{\Psi_\theta}_t)^{-1}(x_t)\right\}, \notag 
 \end{eqnarray} 
 $$\frac{ d\mathcal{L}^\pi_{OFM}}{d \theta} := \frac{ d}{d \theta} \EE_{t; x_0,x_1 \sim \pi}\!\!\left\la  \textsc{NO-GRAD}\left\{ 2 \left( t \nabla^2 \Psi_{\theta}(z_0)  
 +  (1-t) I\right)^{-1} \frac{(x_0 - z_0)}{t} \right\},   \nabla \Psi_\theta(z_0) \right\ra,$$
 where variables under $\textsc{NO-GRAD}$ remain constants during differentiation. 

\end{proposition}

\textbf{Flow map inversion.} In order to find the initial point $z_0 = (\phi^\Psi_t)^{-1}(x_t)$, we note that \eqref{eq:init point eq}
$$x_t = (1-t)  z_0 + t \nabla \Psi(z_0)$$
is equivalent to 
 \begin{eqnarray}
     \nabla \left(\frac{(1-t)}{2} \| \cdot\|^2 + t \Psi (\cdot)  - \la x_t , \cdot \ra \right) (z_0) = 0. \notag
 \end{eqnarray}
The function under gradient operator $\nabla$ has minimum at the required point $z_0$, since at $z_0$ the gradient of it equals $0$. If $ t < 1$ the function is at least $(1-t)$-strongly convex, and the minimum is unique. The case $t = 1$ is negligible in practice, since it has zero probability to appear during training.

We can reduce the problem of inversion to the following minimization subproblem
 \begin{eqnarray}\label{eq:min subtask}
     (\phi^\Psi_t)^{-1}(x_t) = \arg\min_{z_0 \in \R^D} \left[ \frac{(1-t)}{2} \| z_0\|^2 + t \Psi (z_0) - \la x_t , z_0 \ra \right].
 \end{eqnarray} 
Optimization subproblem \eqref{eq:min subtask} is at least \textbf{$(1-t)$-strongly convex} and can be effectively solved  for any given point $x_t$ (in comparison with typical non-convex optimization tasks). %For solving \eqref{eq:min subtask} one can use either accelerated first-order algorithms for strongly convex functions, e.g., Nesterov Accelerated Gradient, or more common solvers \cite{ruder2016overview}.

 
\textbf{Algorithm.}  The Optimal Flow Matching pseudocode is presented in listing \ref{alg:OFM}. We estimate math expectation over plan $\pi$ and time $t$ with uniform distribution on $[0,1]$ via unbiased Monte Carlo.

\begin{algorithm}[ht!]
\caption{\algname{Optimal Flow Matching} }
\label{alg:OFM}   
\begin{algorithmic}[1]
\REQUIRE Initial transport plan $\pi \in \Pi(p_0, p_1)$, number of iterations $K$,  batch size $B$, optimizer $Opt$, sub-problem optimizer $SubOpt$, ICNN $\Psi_\theta$ 
\FOR{$k=0,\ldots, K-1$}
\STATE Sample batch $\{(x^i_0, x^i_1)\}_{i=1}^B$ of size $B$ from plan $\pi$;
\STATE Sample times batch $\{t^i\}_{i=1}^B$ of size $B$ from $U[0,1]$;
\STATE Calculate linear interpolation $x^i_{t^i} = (1-t^i)x^i_0 + t^i x^i_1$ for all $i \in \overline{1,B}$;
\STATE Find the initial points $z^i_0$ via solving the convex problem with $SubOpt$:

$$z^i_0 = \textsc{NO-GRAD} \left\{ \arg\min_{z^i_0} \left[ \frac{(1-t^i)}{2} \| z^i_0\|^2 + t^i \Psi_\theta (z^i_0) - \la x^i_{t^i} , z^i_0 \ra \right]\right \};$$
\STATE Calculate loss $\hat{\mathcal{L}}_{OFM}$ 
$$\hat{\mathcal{L}}_{OFM} = \frac{1}{B} \sum_{i=1}^B \left\la  \textsc{NO-GRAD} \left\{ 2 \left( t^i \nabla^2 \Psi_{\theta}(z^i_0)  
 +  (1-t^i) I \right)^{-1} \frac{(x^i_0 - z^i_0)}{t^i} \right\},   \nabla \Psi_\theta(z^i_0) \right\ra;$$

\STATE Update parameters $\theta$ via optimizer $Opt$ step with $\frac{d \hat{\mathcal{L}}_{OFM}}{d \theta}$;  
\ENDFOR 
\end{algorithmic}
\end{algorithm}


\subsection{Relation to Prior Works}\label{sec:prior}
\vspace{-2mm}
In this subsection, we compare our Optimal Flow Matching and previous straightening approaches. One unique feature of OFM is that it works only with flows which have straight paths by design and does not require ODE integration to transport points. Other methods may result in non-straight paths during training, and they still have to solve ODE even with near-straight paths.   

\textbf{OT Solvers} \cite{taghvaei20192,makkuva2020optimal, amos2023on}. According to Theorem \ref{thm:loss equiv}, our OFM and dual OT solvers basically minimize the same OT loss \eqref{eq:dual formulation}. However, our OFM actively utilizes the temporal component of the dynamic process. It allows us to pave a novel theoretical bridge between OT and FM. Such a direct connection can lead to the adoption of the strengths of both methods and a deeper understanding of them.


\textbf{OT-CFM} \cite{pooladian2023multisample,tong2024improving}. Unlike our OFM approach, OT-CFM method retrieves biased OT solution, and the recovery of straight paths is not guaranteed. In OT-CFM, minibatch OT plan appears as a heuristic that helps to get better trajectories in practice. In contrast, usage of  \textbf{any} initial transport plan $\pi$ in our OFM is completely justified in Theorem \ref{thm:loss equiv}.

\textbf{Rectified Flow} \cite{liu2023flow,liu2022rectified}. In Rectified Flows \cite{liu2023flow}, the authors iteratively apply Flow Matching to refine the obtained trajectories. However, in each iteration, RF accumulates error since one may not learn the exact flow due to neural approximations. In addition, RF does not guarantee convergence to the OT plan for the quadratic cost. The $c$-Rectified Flow \cite{liu2022rectified} modification  can converge to the OT plan for any cost function $c$, but still remains iterative. In addition, RF and $c$-RF both requires ODE simulation after the first iteration to continue training. In OFM, we work only with the quadratic cost function, but retrieve its OT solution in \textbf{just one FM iteration} without simulation of the trajectories. %Finally, unlike RF, OFM does not have a marginal preserving property, except $$ 

\textbf{Light and Optimal Schrödinger Bridge.} In \cite{gushchin2024light}, the authors observe the relation between Entropic Optimal Transport (EOT) \cite{leonard2013survey, chen2016relation} and Bridge Matching (BM) \cite{shi2024diffusion} problems. These are stochastic analogs of OT and FM, respectively. In EOT and BM, instead of deterministic ODE and flows, one considers stochastic processes with non-zero stochasticity. The authors prove that, during BM, one can restrict considered processes only to the specific ones and retrieve the solution of EOT. 

\subsection{Theory: properties of OFM}

In this subsection, we provide the OFM's theoretical properties, which give an intuition for understanding of its main working principles and behavior. 

\begin{proposition}[Simplified OFM Loss]\label{prop:ofm_simplified_loss} We can simplify \eqref{eq:OFM loss} to a more suitable form: 
\begin{equation}\label{eq:OFM loss simple}
    \mathcal{L}^\pi_{OFM}(\Psi)\!\! = \!\!   \int\limits_0^1 \left \{  \int\limits_{\R^D \times \R^D} \left|\left|\frac{(\phi^\Psi_t)^{-1} (x_t) - x_0 }{t}\right|\right|^2 \pi(x_0, x_1) dx_0 dx_1\right\} dt, x_t = (1-t)  x_0 + t x_1.
\end{equation}
\end{proposition}
The simplified form \eqref{eq:OFM loss simple} shows that OFM loss actually measures how well $\Psi$  restores initial points $x_0$ of linear interpolations depending on future point $x_t$ and time $t$. 

\textbf{Generative properties of OFM.}
In this paragraph, we provide another view on our OFM approach. In our OFM, we aim to construct a vector field $u$ which is as close to the dynamic OT field $u^*$ as possible. We can use the least square regression to measure the distance between them:
\begin{equation}\label{eq:dist}
    \textsc{dist}(u, u^*) := \int_{0}^1 \int_{\R^D} \|u_{t} (x_t) - u_t^* (x_t)\|^2\underset{:= p^*_t(x_t)}{\underbrace{\phi_t^*\#p_0(x_t)}} dx_t dt.
\end{equation} % We also fix any initial plan $\pi$. 
\begin{proposition}[Intractable Distance]\label{prop:intractable dist}
    The distance $\textsc{dist}(u, u^*)$  between an arbitrary vector field $u$ and OT field $u^* $ equals to the FM loss from \eqref{eq:rectflow min} with the optimal plan $\pi^*$, i.e., 
    $$\textsc{dist}(u, u^*) = \mathcal{L}^{ \pi^*}_{FM}(u) -  \underset{=0}{\underbrace{\mathcal{L}^{\pi^*}_{FM}(u^*)}}.$$
\end{proposition}
\vspace{-2mm}
We can not minimize intractable $\textsc{dist}(u, u^*)$ since the optimal plan $\pi^*$ is unknown. In OT-CFM \cite{tong2024improving}, authors heuristically approximate $\pi^*$ in $\mathcal{L}_{FM}^{\pi^*}(u)$, but obtain biased solution. Surprisingly, for the \textit{optimal} vector fields, the distance can be calculated explicitly via \textbf{any} known plan $\pi$.


\begin{proposition}[Tractable Distance For OFM]\label{prop:dist for optimal}
    The distance $\textsc{dist}(u^{\Psi}, u^{\Psi^*})$ between  an \textbf{optimal} vector field $u^\Psi$ generated by a convex function $\Psi$ and the vector field  $u^{\Psi^*}$ with the Brenier potential $\Psi^*$ can be evaluated directly via OFM loss  \eqref{eq:OFM loss} and \textbf{any} plan $\pi$: 
    \begin{equation}
   \textsc{dist}(u^{\Psi}, u^{\Psi^*}) =  \mathcal{L}_{FM}^\pi(u^\Psi) - \mathcal{L}^\pi_{FM}(u^{\Psi^*}) = \mathcal{L}_{OFM}^\pi(\Psi) - \mathcal{L}^\pi_{OFM}(\Psi^*). \label{eq:fields diff bound}
    \end{equation}
\end{proposition}
\vspace{-1mm}
In \eqref{eq:fields diff bound}, the first term is our tractable OFM loss, and the second term does not depend on $\Psi$. Hence, during the whole minimization process in our OFM, we gradually lower the distance \eqref{eq:dist} between the current vector field and the dynamic OT field up to the complete match.

\vspace{-3mm}
