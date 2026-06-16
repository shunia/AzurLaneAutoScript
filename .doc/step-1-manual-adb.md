# 第 1 步：以手动 ADB 连接作为输入前提

## 目标

确认 macOS 迁移验证可以从一个用户显式提供的 ADB serial/address 开始。

本阶段只建立后续截图验证的输入前提，不验证截图内容、图像识别、OCR、任务调度或 WebUI。

## 输入

用户提供一个 ADB 目标，例如：

```text
127.0.0.1:5555
emulator-5554
R5CTxxxxxxx
```

## 实施范围

本阶段采用手动 ADB 连接，不引入自动设备发现、模拟器识别或模拟器管理。

实施时应优先沿用项目现有设备配置字段承载 serial/address，例如 `Emulator_Serial` 所在配置项。后续截图链路应从该配置读取目标设备，而不是依赖 Windows 模拟器自动发现逻辑。

## 执行步骤

1. 在 macOS 环境安装并确认 `adb` 可执行。
2. 用户手动连接目标设备或模拟器。
3. 用户将目标 serial/address 写入项目配置。
4. 执行 ADB 可达性检查。
5. 记录本阶段验证信息，作为后续截图验证的输入说明。

## 验证方式

执行：

```bash
adb devices
adb -s <serial> get-state
adb -s <serial> shell echo alas-adb-ok
```

通过标准：

```text
adb -s <serial> get-state
```

返回：

```text
device
```

并且：

```text
adb -s <serial> shell echo alas-adb-ok
```

返回：

```text
alas-adb-ok
```

## 记录信息

验证通过后记录以下信息：

- macOS 版本
- CPU 架构：Intel 或 Apple Silicon
- ADB 版本
- ADB serial/address
- 设备来源：真机、模拟器或云手机
- 游戏服务器：CN、EN、JP 或 TW

## 退出条件

指定 serial/address 可以稳定执行基础 ADB shell 命令后，进入第 2 步“获取真实游戏截图”。

## 失败处理

- `unauthorized`：在设备上授权 ADB 后重试。
- `offline`：重连 ADB，必要时重启设备侧 ADB。
- `not found`：检查 serial/address，必要时重新执行 `adb connect <address>`。
- 多设备冲突：必须显式指定 serial，不在本阶段实现自动选择。

## 非目标

- 不实现自动设备扫描。
- 不实现模拟器自动发现。
- 不实现模拟器自动启动或停止。
- 不实现多设备选择 UI。
- 不验证截图内容、OpenCV、OCR 或任务执行。
