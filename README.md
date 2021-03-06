# RayTracingCUDA

Project was created in order to implement rendering ray traced images with CUDA library.

First implementation was created without GPU computing as I wanted to properly understand how ray tracing works. For this I have followed Peter Shirley Ray Tracing In One Weekend course. It is placed in [RayTracingInOneWeekendWithoutCUDA](https://github.com/Mregussek/RayTracingCUDA/tree/main/RayTracingInOneWeekendWithoutCUDA) directory. Rendered images with CPU are placed in [RayTracingInOneWeekendWithoutCUDA/img](https://github.com/Mregussek/RayTracingCUDA/tree/main/RayTracingInOneWeekendWithoutCUDA/img).

Afterwards, main objective was to understand correctly CUDA computing library and use it to rewrite first implementation. Results are stored in [ActualProject](https://github.com/Mregussek/RayTracingCUDA/tree/main/ActualProject) directory. Also I have added loading scenes from json files using [nlohmann/json](https://github.com/nlohmann/json) library. Example json scenes are stored in [ActualProject/resources](https://github.com/Mregussek/RayTracingCUDA/tree/main/ActualProject/resources) catalog. Rendered images with GPU CUDA computing are placed in [ActualProject/img](https://github.com/Mregussek/RayTracingCUDA/tree/main/ActualProject/img).

## Rendered images with CUDA

<p align="center">
  <img width="960" height="526" src="ActualProject/img/default_json.png">
</p>

<p align="center">
  <img width="960" height="526" src="ActualProject/img/second_json.png">
</p>

<p align="center">
  <img width="960" height="526" src="ActualProject/img/third_json.png">
</p>

## Terminal prints

<p align="center">
  <img width="563" height="641" src="ActualProject/img/terminal_results.png">
</p>

## Appendix

[Ray Tracing in One Weekend Peter Shirley](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
[Accelerated Ray Tracing in One Weekend in CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)
[nlohmann/json](https://github.com/nlohmann/json)
