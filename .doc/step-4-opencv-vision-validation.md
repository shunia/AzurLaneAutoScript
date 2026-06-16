# 第 4 步：验证 OpenCV 图像识别

## 目标

对第 3 步产出的标准截图运行 Alas 现有 OpenCV 识别逻辑，生成诊断报告。

本阶段只报告识别事实，由人结合截图判断识别是否合理。本阶段不建立自动 expected 断言，不修改现有 OpenCV/OCR 业务接口。

## 输入

第 3 步输出的标准截图文件，例如：

```text
normalized_screenshot.png
```

以及第 3 步记录的截图元数据。

## 输出

一份 OpenCV 诊断报告。

报告用于展示现有识别接口在该截图上的返回结果，例如页面候选、按钮匹配结果、模板相似度、多目标匹配数量和耗时。

报告不给出“是否符合人工预期”的最终结论。人打开截图和报告后自行判断识别是否合理。

## 实施范围

本阶段实现一个只读诊断流程：

1. 读取第 3 步产出的标准截图。
2. 调用项目现有 OpenCV 识别接口。
3. 记录接口返回值、耗时和必要的定位信息。
4. 输出报告供人审阅。

本阶段不能改变以下现有接口的返回结构：

- `Button.appear_on(image)` 返回 `bool`
- `Button.match(image, offset=...)` 返回 `bool`
- `Button.match_binary(image, offset=...)` 返回 `bool`
- `Button.match_luma(image, offset=...)` 返回 `bool`
- `Template.match(image, ...)` 返回 `bool`
- `Template.match_result(image, ...)` 返回 `(similarity, Button)`
- `Template.match_multi(image, ...)` 返回 `list[Button]`
- `image_color_count(...)` 返回 `bool`

如诊断报告需要额外字段，应由诊断工具包装记录，不能要求业务接口输出 JSON 或改变返回值。

## 执行流程

### 1. 读取标准截图

使用 OpenCV 读取第 3 步产物：

```python
import cv2

image = cv2.imread("normalized_screenshot.png")
assert image is not None
```

本阶段不再旋转、裁切、resize 或转换截图输入；这些都属于第 3 步职责。

### 2. 自动页面候选诊断

遍历 `module.ui.page.Page` 中的页面检查按钮。

对每个有效的 `page.check_button` 调用现有按钮匹配接口，例如：

```python
page.check_button.match(image, offset=(30, 30))
```

报告记录每个页面候选的 `page.name`、按钮名和 `bool` 返回值。

### 3. 常用 UI 按钮诊断

扫描常用 UI 资产，记录它们在当前截图上的匹配结果。

优先覆盖：

- `MAIN_GOTO_FLEET`
- `MAIN_GOTO_CAMPAIGN`
- `MAIN_GOTO_CAMPAIGN_WHITE`
- `MAIN_GOTO_REWARD`
- `GOTO_MAIN`
- `BACK_ARROW`
- `CAMPAIGN_CHECK`
- `COMMISSION_CHECK`
- `RESEARCH_CHECK`
- `SHOP_REFRESH_CHECK`

调用方式保持现有项目语义，例如：

```python
button.match(image, offset=(30, 30))
```

或在适合固定颜色判断的场景调用：

```python
button.appear_on(image)
```

### 4. 通用 Template 诊断

对有代表性的 `Template` 资源执行诊断。

优先覆盖：

- `TEMPLATE_FORMATION_1`
- `TEMPLATE_FORMATION_2`
- `TEMPLATE_FORMATION_3`
- `TEMPLATE_AIR_STRIKE_ICON`
- `TEMPLATE_MOB_MOVE_ICON`
- `TEMPLATE_MANJUU`
- `TEMPLATE_RUNNING`
- `TEMPLATE_WAITING`
- `TEMPLATE_DETAIL`

对只需 bool 的模板，调用：

```python
template.match(image)
```

对需要定位相似度的模板，调用：

```python
similarity, button = template.match_result(image)
```

报告记录 `similarity` 和 `button.area`。

### 5. 多目标匹配诊断

对使用 `Template.match_multi()` 的场景，报告返回数量和匹配区域。

优先覆盖：

- `TEMPLATE_MANJUU.match_multi(...)`
- 商店货币图标模板
- 战役章节标记模板

报告记录：

- 模板名
- 返回数量
- 每个返回 `Button.area`

### 6. 颜色状态诊断

对现有颜色判断路径做少量探测。

优先覆盖项目中已有的 `image_color_count(...)` 使用场景，例如商店刷新状态、科研队列状态或自动搜索状态。

报告记录原始 `bool` 返回值。若需要像素计数，只能由诊断工具额外计算并记录，不能修改 `image_color_count(...)` 的返回值。

## 建议诊断顺序

诊断工具按以下优先级输出结果：

1. P0 页面候选：`Page` 检查按钮
2. P1 常用 UI 按钮：主界面、返回、页面 check
3. P2 通用 `Template.match`
4. P3 颜色状态判断
5. P4 `Template.match_multi`
6. P5 地图与复杂视觉

P5 只在有地图截图时执行，不作为本阶段基础诊断的必需项。

## 报告内容

报告至少包含：

- 截图路径
- OpenCV 版本
- Numpy 版本
- 页面候选结果
- 常用 UI 按钮结果
- Template 匹配结果
- 多目标匹配结果
- 颜色状态诊断结果
- 每类诊断耗时
- 异常和失败堆栈

报告示例：

```text
Image: normalized_screenshot.png
OpenCV: 4.x

Page candidates:
- page_main: true
- page_commission: false
- page_research: false

UI buttons:
- MAIN_GOTO_FLEET: true
- MAIN_GOTO_REWARD: true
- BACK_ARROW: false
- COMMISSION_CHECK: false

Template results:
- TEMPLATE_FORMATION_1: false
- TEMPLATE_DETAIL: similarity=0.42, area=(100, 200, 150, 240)

Multi-template:
- TEMPLATE_MANJUU: count=0
```

## 人工判断方式

人打开标准截图和报告，对照判断：

- 当前截图看起来是什么页面。
- 页面候选是否合理。
- 可见按钮是否被识别。
- 不可见按钮是否出现误报。
- 模板相似度是否存在明显异常。
- 多目标匹配数量和区域是否符合画面。

人工判断结果不在本阶段固化为 expected。只有进入“建立截图样本和识别基准”阶段后，才把人工确认过的结论沉淀为回归基准。

## 退出条件

本阶段完成后应具备：

- 至少一张标准截图的 OpenCV 诊断报告。
- 报告包含页面候选、常用 UI 按钮和至少一类 Template 结果。
- 人可以通过报告定位误报、漏报或版本差异。
- 没有修改现有 OpenCV 业务接口返回结构。

## 失败处理

- 页面候选全部为 false：回到第 3 步检查截图方向、尺寸、颜色通道和服务器资源。
- 多个页面候选为 true：记录为候选冲突，由人结合截图判断是否误报。
- 可见按钮为 false：记录按钮名、截图和调用参数。
- 不可见按钮为 true：记录误报按钮名和截图区域。
- 模板相似度异常：记录 OpenCV 版本、模板名、相似度和匹配区域。
- 地图复杂识别失败：单独归类为复杂视觉问题，不阻塞基础 UI 诊断。

## 非目标

- 不验证 OCR。
- 不获取截图。
- 不归一化截图。
- 不执行点击或滑动。
- 不运行 Alas 任务。
- 不启动 WebUI。
- 不设计自动 expected 断言。
- 不修改 OpenCV 识别阈值。
- 不修改现有业务接口返回结构。
