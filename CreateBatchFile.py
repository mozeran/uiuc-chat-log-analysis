# A temporary plan to adapt phrase-LDA
def write(num_topics=10, min_support=30, max_pattern=3, iter_times=1000):
    """
    Replace several parameters in the batch commands.
    :param num_topics: number of topics to extract, same as LDA
    :param min_support: minumum support (minimum times a phrase candidate should appear in the corpus to be significant)
    :param max_pattern: max size you would like a phrase to be (if you don't want too long of phrases that occasionally occur)
    :param iter_times: times of iteration
    """
    with open('win_run_default.bat', 'r') as default, open('win_run.bat', 'w') as updated:
        commands = default.read().splitlines()
        for line in commands:
            if line.startswith('set numTopics'):
                new_line = 'set numTopics='+str(num_topics)
                commands[commands.index(line)] = new_line
            if line.startswith('@set minsup'):
                new_line = '@set minsup='+str(min_support)
                commands[commands.index(line)] = new_line
            if line.startswith('@set maxPattern'):
                new_line = '@set maxPattern=' + str(max_pattern)
                commands[commands.index(line)] = new_line
            if line.startswith('@set gibbsSamplingIterations'):
                new_line = '@set gibbsSamplingIterations=' + str(iter_times)
                commands[commands.index(line)] = new_line
        print('\n'.join(commands), file=updated)
