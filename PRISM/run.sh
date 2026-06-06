for dataset in 'citeseer' 'dblp' 'cora' 'arxiv' 'pubmed'; do
  for init_n_per_class in 3 5 10; do
    for perspective in 3 5 10; do
      echo "python main.py --device \"cuda:0\" --dataset $dataset --init_n_per_class $init_n_per_class --perspective $perspective" >> $QUEUE_FILE
    done
  done
done

TOTAL_JOBS=$(wc -l < $QUEUE_FILE)
echo "✅ Queue creation completed! A total of $TOTAL_JOBS experiments are waiting."
echo "🚀 Starting parallel execution..."

CONCURRENT_JOBS=18

xargs -a "$QUEUE_FILE" -I CMD -P "$CONCURRENT_JOBS" bash -c 'echo "▶️ [Running] CMD"; CMD'

echo "🎉 All parallel experiments have been completed!"