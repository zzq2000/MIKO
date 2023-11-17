1. chat-keyinfo.py 是主程序，按照顺序执行关键信息的生成和intention(内置prompt生成功能，可以根据关键信息生成对应的intention获取prompt)的生成两个步骤，在使用的时候根据注释修改输入输出地址即可

Note：其中test(train/valid)2015(2017)_keyinfo.json是输入文档，他包含了5个字段--[1] "question_id"(这个不用管)   [2]"image_id"(图像的名称)  [3]"text"(原始贴文信息)
[4] "image_descrption"(llava生成的图像描述)     [5]"prompt"（引导gpt3.5获取关键信息的prompt）



2. 运行环境：python>=3.8.1, openai>=0.28.0



3. 实际运行费用，目前已测试了20个样本按照顺序执行步骤2（key-info获取）和步骤3（3-5个intention的生成），目前花费（0.20RMB），预计总开销（0.20*10000=2000RMB）