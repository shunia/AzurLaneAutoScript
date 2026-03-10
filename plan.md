# AzurLaneAutoScript 跨平台改造计划（状态同步版）

- 更新时间：2026-03-06
- 同步依据：当前工作区代码（含未提交改动）
- 当前工作区相关变更：
  - `M module/ocr/al_ocr.py`
  - `?? requirements-cross-platform.txt`
  - `?? .github/workflows/build.yml`
  - `?? plan.md`

## 状态定义

- `✅` 已完成
- `🟡` 部分完成
- `⬜` 未开始
- `⛔` 需验证

---

## 0. 项目现状总览

1. 平台入口已分流：非 Windows 默认走 `PlatformBase`，避免直接加载 `PlatformWindows`。
2. OCR 已改为双后端框架（MXNet + ONNX）雏形，但依赖和模型路径策略尚未闭环。
3. 截图方法在非 Windows 上已对 `nemu_ipc/ldopengl` 做回退。
4. 设备自动发现和进程管理仍以 Windows 逻辑为核心，跨平台实现未落地。
5. 多平台 GitHub Actions 打包流程已有草案，但未验证成功。

---

## 1. 阶段一：解除平台锁定（必须完成）

### 1.1 OCR 引擎替换

状态：`🟡 部分完成`

已同步现状：
- `module/ocr/al_ocr.py` 已改成运行时检测后端：优先 `mxnet`，否则使用 `onnxruntime + CnOcr`。
- 保留旧模型兼容逻辑（MXNet 分支），并新增 ONNX 分支。

未完成项：
- `requirements.txt` 仍是 `cnocr==1.2.2` + `mxnet==1.6.0`，尚未替换为 ONNX 方案。
- `requirements-cross-platform.txt` 新增 `onnxruntime`，但缺少 `cnocr[ort-cpu]` 依赖声明。
- `al_ocr.py` 中 `_get_model_dir()` 已定义但未接入实际加载流程。

本阶段完成标准：
- 依赖文件与 OCR 运行时逻辑一致（至少一套可复现安装路径）。
- 三平台 OCR smoke test 可通过（中/英/日/繁关键场景）。

---

### 1.2 条件导入修复

状态：`🟡 部分完成`

已同步现状：
- `module/device/platform/__init__.py` 已按 `IS_WINDOWS` 分流 `PlatformWindows` / `PlatformBase`。
- `module/device/device.py` 中，非 Windows 选择 `nemu_ipc` / `ldopengl` 会自动回退 `auto`。
- `module/device/method/nemu_ipc.py` 与 `module/device/method/ldopengl.py` 已有 `IS_WINDOWS` 可用性判断。

未完成项：
- `module/device/screenshot.py` 与 `module/device/control.py` 仍静态混入 Windows 特性类，主要依赖运行时判断兜底。
- `module/device/platform/emulator_windows.py` 仍顶层 `import winreg`（当前通过平台分流避免非 Windows 导入，但缺少单元级导入验证）。

本阶段完成标准：
- 在 macOS/Linux 上可完成全量模块导入（至少关键入口模块）且不报平台相关 ImportError。

---

### 1.3 截图方法回退

状态：`🟡 部分完成`

已同步现状：
- 非 Windows 上已禁用 `nemu_ipc` / `ldopengl`（配置层面回退到 `auto`）。

未完成项：
- 计划中的默认优先级 `Scrcpy > DroidCast > uiautomator2 > ADB` 未落地。
- `run_simple_screenshot_benchmark()` 当前候选不包含 `scrcpy`，与计划目标不一致。

本阶段完成标准：
- macOS/Linux 连接设备后能稳定完成截图，并在日志中可见预期回退链路。

---

## 2. 阶段二：基础可用性（重要）

### 2.1 设备发现降级

状态：`🟡 部分完成`

已同步现状：
- `module/device/platform/emulator_base.py`、`platform_base.py` 已存在，跨平台基类框架已就位。

未完成项：
- `EmulatorManagerBase` 目前仍返回空列表，尚未实现 `adb devices` 扫描。
- 非 Windows 的自动发现能力仍不足，主要依赖手动配置序列号。

本阶段完成标准：
- 非 Windows 至少支持 `adb devices` 自动扫描并回填候选设备。

---

### 2.2 进程管理跨平台化

状态：`⬜ 未开始`

现状：
- `execute()` / `kill_process_by_regex()` 只在 `PlatformWindows` 中实现。
- 尚无 `platform_linux.py`、`platform_macos.py`。

本阶段完成标准：
- 跨平台进程管理接口统一到 `PlatformBase` 抽象并具备 Linux/macOS 实现。

---

## 3. 阶段三：体验优化（可选）

### 3.1 窗口管理跨平台实现

状态：`⬜ 未开始`

### 3.2 模拟器自动发现（macOS）

状态：`⬜ 未开始`

### 3.3 模拟器启动/停止（macOS）

状态：`⬜ 未开始`

---

## 4. GitHub Actions 配置

### 4.1 多平台打包 Workflow

状态：`🟡 部分完成（文件已新增，未验证）`

已同步现状：
- 已新增 `.github/workflows/build.yml`。
- 已包含 Windows/macOS/Linux 三平台矩阵和打包上传流程。

未完成项：
- 尚未跑通 CI 验证。
- Windows 下 PyInstaller `--add-data` 分隔符可能与平台语法不一致，存在失败风险。
- 依赖安装依赖 `requirements-cross-platform.txt`，而该文件尚未完全闭环 OCR 依赖。

---

## 5. 测试验证清单（按当前状态更新）

### 阶段一验证
- [ ] Windows OCR（MXNet 路径）回归通过
- [ ] macOS OCR（ONNX 路径）通过
- [ ] Linux OCR（ONNX 路径）通过
- [ ] macOS/Linux 关键模块导入测试通过
- [ ] macOS/Linux 截图 fallback 链路通过

### 阶段二验证
- [ ] macOS 手动配置设备可稳定运行
- [ ] Linux 手动配置设备可稳定运行
- [ ] `adb devices` 自动扫描实现并通过三平台测试
- [ ] 三平台进程管理实现并通过测试

### CI 验证
- [ ] Windows 打包产物可生成并可启动
- [ ] macOS 打包产物可生成并可启动
- [ ] Linux 打包产物可生成并可启动

---

## 6. 下一步执行顺序（按阻塞优先级）

1. 先收敛 OCR 依赖策略（`requirements.txt` 与 `requirements-cross-platform.txt` 统一）。
2. 为非 Windows 增加 `adb devices` 发现实现，补齐阶段二最小可用闭环。
3. 明确 `auto` 截图优先级并把 `scrcpy` 纳入候选链路（如可用）。
4. 修正并验证 `build.yml` 在三平台 runner 的实际可执行性。

