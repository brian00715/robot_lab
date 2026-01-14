#!/bin/bash
# 杀死所有Isaac Lab相关的死机进程

echo "正在查找Isaac Lab进程..."
PIDS=$(ps aux | grep -E "(isaac|omni|python.*(play|train)\.py)" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "✅ 没有发现运行中的Isaac Lab进程"
else
    echo "发现以下进程："
    ps aux | grep -E "(isaac|omni|python.*(play|train)\.py)" | grep -v grep | awk '{printf "  PID: %s  CPU: %s%%  MEM: %s%%  CMD: %s\n", $2, $3, $4, substr($0, index($0,$11))}'
    
    echo ""
    read -p "是否杀死这些进程? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在杀死进程..."
        kill -9 $PIDS
        echo "✅ 进程已杀死"
        
        # 再次检查
        sleep 1
        REMAINING=$(ps aux | grep -E "(isaac|omni|python.*(play|train)\.py)" | grep -v grep | wc -l)
        if [ $REMAINING -eq 0 ]; then
            echo "✅ 所有进程已成功清理"
        else
            echo "⚠️  仍有残留进程，请手动检查"
        fi
    else
        echo "❌ 已取消"
    fi
fi
