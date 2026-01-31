# 一般公式
$$ C_{in}\frac{dT_{in}}{dt} = Q_{in} + I_{0}\sin(\theta_1)\cos(\theta_2 - \gamma)(\eta S) \tau (1 - F_{shade}) + u_w(\eta S)(T_{out} - T_{in}) + h_{in}(1 - \eta) S (T_{w}(D, t) - T_{in}) $$

**参数说明：**

| 符号        | 含义           |
| :---------- | :------------- |
| $C_{in}$    | 比热           |
| $Q_{in}$    | 内部热量       |
| $I_0$       | 光强           |
| $\theta_1$  | 高度角         |
| $\theta_2$  | 方位角         |
| $\gamma$    | 窗户方位角     |
| $\eta$      | 窗墙比         |
| $S$         | 墙面积         |
| $\tau$      | 窗户太阳透射率 |
| $F_{shade}$ | 遮挡比         |
| $u_w$       | 窗户传热系数   |
| $h_{in}$    | 热阻           |

# 补充公式
$$ \rho_{w}c_{w}\frac{\partial T_{w}}{\partial t} = k\frac{\partial^2 T_{w}}{\partial x^2}, x\in[0, D] $$

上述方程（墙体一维非稳态导热微分方程）的边界条件（限制条件）如下：

$$
\begin{cases}
-k\frac{\partial T}{\partial x}\bigg|_{x=0} = h_{out}(T_{out} + k_{const}I_{0}\sin(\theta_1)\cos(\theta_2 - \gamma) - T_w(0, t)) \\
-k\frac{\partial T}{\partial x}\bigg|_{x=D} = h_{in}(T_w(D, t) - T_{in}(t))
\end{cases}
$$

**参数说明：**

| 符号        | 含义                                                        |
| :---------- | :---------------------------------------------------------- |
| $T_w(D, t)$ | 墙体内表面温度                                              |
| $\rho$      | 密度                                                        |
| $c$         | 比热                                                        |
| $k$         | 导热系数（公式中的第一个 $k$）                              |
| $k_{const}$ | 常系数（公式中的第二个 $k$，此处记为 $k_{const}$ 以示区别） |