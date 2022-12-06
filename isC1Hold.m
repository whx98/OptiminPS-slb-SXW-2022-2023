function flag = isC1Hold(G,NodeBranchMat,LineNum,Pimax,Qimax,vmin,headNode,endNode,r,x)
temp = table2array(G.Edges);
num = zeros(size(NodeBranchMat,1),1);
for i = 1:size(num,1)
    num(i,1) = size(find(temp==i),1);
end
LeafNodes=find(num==1);
LeafNodes=LeafNodes(2:end); % delete node 1
% calculate hat
link = zeros(LineNum,LineNum);
for i = 1:LineNum
    link(i,i) = -1;
end
for i = 1:LineNum
    code = i+1;
    endcode = [];
    for j = 1:LineNum
        for k = 1:size(code,1)
            if isempty(find(temp(:,1)==code(k)))
                k = k+1;
            end
            if k > size(code,1)
                break;
            end
            linenum = find(temp(:,1)==code(k));
            for m = 1:size(linenum,1)
                endcode = [endcode;temp(linenum(m),2)];
                link(i,temp(linenum(m),2)-1) = -1;
            end
        end
        code = endcode;
        endcode = [];
    end
end
Pij_hat = abs(link(:,:)*Pimax(2:end));
Qij_hat = abs(link(:,:)*Qimax(2:end));
IJ=[headNode endNode];
Al = cell(length(LeafNodes),1);
PathL = cell(length(LeafNodes),1);
uij=zeros(length(LeafNodes),2);
for i=1:length(LeafNodes)
    lt=LeafNodes(i);
    Path = shortestpath(G,lt,1);
    Al{i}=zeros(2,2,length(Path)-1);
    for j=1:length(Path)-1 % for all node in Path from ?, calculate A_l
%         nl=[]; % the set of 1,2,...,nl in C1
        for k=1:LineNum
            if (IJ(k,1)==Path(j) && IJ(k,2)==Path(j+1))||(IJ(k,2)==Path(j) && IJ(k,1)==Path(j+1))
                PathL{i}=[PathL{i},Path(j+1)];
                Al{i}(:,:,j)=diag([1 1])-2/vmin*mtimes([r(k);x(k)],[Pij_hat(k),Qij_hat(k)]);
            end
        end
    end
    uijtemp=diag([1 1]);
    for m=2:length(Path)-1
        uijtemp=uijtemp*Al{i}(:,:,m);
    end
    uij(i,:)=(uijtemp*[r(find(endNode==lt));x(find(endNode==lt))])';
end
if find(uij<0)
    flag = 0;
else
    flag = 1;
end
end