
local function extractWeightLayers(modules, prev)
   local layers = {}
   local prev = prev or nil
   for k,v in pairs(modules) do
      if v.modules and torch.typename(v) ~= 'nn.WeightNorm' then
         for k2,v2 in pairs(extractWeightLayers(v.modules, prev)) do
            table.insert(layers, v2)
         end
      elseif v.weight and v.weight:dim() == 4 then
         table.insert(layers, v)
      end
      prev = v
   end
   return layers
end

local function genOrthogonal(dim)
    local a = torch.Tensor(dim, dim):normal(0, 1)
    local q, r = torch.qr(a)
    local d = torch.diag(r):sign()

    local diagsize = d:size(1)
    local d_exp = d:view(1, diagsize):expand(diagsize, diagsize)

    q:cmul(d_exp) --make uniform
    return q
end

local function makeDeltaOrthogonal(weights)
    if weights:size(2) > weights:size(1) then
        print("channels_in greater than channels_out")
        --return
    end
    weights:zero()
    local dim = math.max(weights:size(1), weights:size(2))
    local q = genOrthogonal(dim)
    local mid1 = math.floor(weights:size(3) / 2) + 1
    local mid2 = math.floor(weights:size(4) / 2) + 1
    weights[{{},{},mid1,mid2}] = q[{{1,weights:size(1)}, {1,weights:size(2)}}]
end

local function initAll(model)
    local layers = extractWeightLayers(model.modules)
    for k, layer in pairs(layers) do
        makeDeltaOrthogonal(layer.weight)
    end
end

