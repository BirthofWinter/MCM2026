# 模型通用性与多地理适应性 (Model Generalization & Adaptability)

我们的模型并非局限于 Sungrove 和 Borealis 这两个特定地点。通过将**物理机理（Latitude-dependent Physics）**与**环境参数（Climate-dependent Parameters）**解耦，该模型可以广泛应用于其他具有相似纬度但气候特征迥异的地区。

## 1. 纬度相似性与气候差异悖论 (Latitude $\ne$ Climate)

在建筑设计中，常见的误区是“同一纬度采用相同设计”。我们的模型分析表明，以下三个关键地理因子会彻底改变被动式策略的有效性，即使太阳路径（Sun Path）完全相同：

1.  **海拔高度 (Altitude)**: 每上升 1000米，气温约下降 6.5°C，同时紫外线辐射增强。
2.  **云量与直射比 (Clearness Index)**: 决定了是被动采暖（利用直射光）还是被动保温（防止散失）。
3.  **昼夜温差 (Diurnal Swing)**: 决定了热质量（Thermal Mass）是否有效。

---

## 2. 案例研究A：热带高地的“供暖需求” (High-Altitude Tropics)
*Sungrove (Lat 1.35°) 的变体*

我们将 Sungrove 的气候数据进行了修正（$T_{amb} - 10^\circ C$, $Radiation \times 1.1$），模拟类似基多（Quito）或内罗毕（Nairobi）等热带高原城市。

*   **模拟结果 (`generalization_altitude_effect.png`)**：
    *   **现象**：虽然中午仍然有强烈的太阳辐射（需要遮阳），但夜间温度骤降至舒适区以下（需要供暖）。
    *   **策略转变**：模型揭示了从 Sungrove 的“全天排热（Cooling Only）”向“日间防晒 + 夜间蓄热（Reject & Store）”策略的转变。
    *   **设计建议**：在平原热带，轻质结构好（散热快）；但在高地热带，必须引入**高热质量墙体**，吸收白天过剩的辐射热，并在寒冷的夜晚缓慢释放，维持室内温度在 20°C 以上。

## 3. 案例研究B：寒带的“阳光红利” (Continental vs. Maritime)
*Borealis (Lat 60.5°) 的变体*

我们将 Borealis (挪威，海洋性气候，多云) 与同纬度的“大陆性气候”（如加拿大草原省份，寒冷但晴朗）进行对比。通过增加直射辐射系数（$Gb \times 2.5$）并降低气温（$T_{amb} - 5^\circ C$）。

*   **模拟结果 (`generalization_cloud_effect.png`)**：
    *   **现象**：在海洋性气候（Borealis Baseline）下，由于云层遮挡，南向大窗户得热有限，主要充当散热面。但在大陆性气候变体中，尽管室外更冷（-10°C），强烈的直射阳光仍能将室内温度被动提升至 20°C 以上。
    *   **策略转变**：
        *   **Borealis (多云)**：策略应为**“超级保温” (Super-Insulation)**。窗户应尽量小且需三层玻璃，减少热散失是第一要务。
        *   **Continental (晴朗)**：策略应为**“被动太阳能采暖” (Passive Solar Heating)**。鼓励南向大开窗 + 深色蓄热地板，利用宝贵的“阳光红利”覆盖供暖负荷。

---

## 4. 全球应用设计指南 (Universal Design Guidelines)

基于上述模拟扩展，我们提出以下针对不同气候的修正因子（Modification Factors）：

| 影响因子     | 对应模型变量调整                  | 推荐设计策略                                                                 | 适用城市示例                 |
| :----------- | :-------------------------------- | :--------------------------------------------------------------------------- | :--------------------------- |
| **高海拔**   | 降低 $T_{out}$，增加 $I_{solar}$  | **增加蓄热质量**。虽然纬度低，但也需要防止夜间过冷。遮阳板需更深（防强UV）。 | 基多 (Quito), 昆明           |
| **高湿度**   | 限制 $Q_{vent}$ 潜能*             | **除湿优先**。自然通风可能失效，需依赖机械除湿或大挑檐防雨。                 | 新加坡, 雅加达               |
| **强大陆性** | 增加 $Gb(i)$，增大昼夜 $\Delta T$ | **最大化南向开窗**。利用晴朗冬季的直射光，配合厚重材质蓄热。                 | 卡尔加里 (Calgary), 乌鲁木齐 |
| **海洋性**   | 降低 $Gb(i)$，减小昼夜 $\Delta T$ | **极致保温**。由于缺乏直射光，窗户主要作为采光而非集热，需控制窗墙比。       | 卑尔根 (Bergen), 伦敦        |

*\*注：当前热模型主要计算显热。对于高湿地区，建议在未来工作中结合焓值（Enthalpy）计算以更准确评估潜热负荷。*

---

> **Discussion Conclusion:**
> "The adaptability of our model lies in its ability to decouple geometric inputs (latitude-based solar angles) from meteorological inputs (weather files). As demonstrated in the generalization study (`codes/generalization_analysis.py`), the optimal passive strategy shifts dramatically even at the exact same latitude. A proactive design must therefore consider not just the sun's position, but the specific 'clearness' of the sky and the 'thermal swing' of the local terrain."
