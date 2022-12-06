function y = myincidence(startNode,endNode)
uniNode = unique([startNode;endNode]);
orderedNode= sort(uniNode);
y = zeros(size(uniNode,2),size(startNode,1));
for i = 1:size(startNode,1)
    y(orderedNode==startNode(i,1),i) = 1;
    y(orderedNode==endNode(i,1),i) = -1;
end
end