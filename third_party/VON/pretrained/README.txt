将训练得到的 args.json 与 epoch-*.pt 放入本目录下任意子目录，例如：

  pretrained/my_run/args.json
  pretrained/my_run/epoch-200.pt

然后设置环境变量（在 SmartFlow 仓库根目录执行）：

  export SMARTFLOW_VON_PRETRAINED=third_party/VON/pretrained/my_run

并重启后端。前端即可调用 POST /von/order 进行排序。
