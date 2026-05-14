# Shared Memory Bank Conflict

这个项目计划用小实验说明：

- shared memory 为什么快
- 什么是 bank
- 什么是 bank conflict
- 为什么访问模式不对时，shared memory 也会变慢

计划内容：

- 写一个无冲突访问版本
- 写一个有冲突访问版本
- 对比不同 stride 下的耗时
- 补充 shared memory 的基本映射解释

目标：

建立“shared memory 不是天然免费，访问方式同样重要”的认识。
