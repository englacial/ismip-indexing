-- ISMIP6 Viewer Quarto Shortcode
--
-- Embeds the ISMIP6 ice sheet model viewer as an iframe.
--
-- Usage:
--   {{< ismip6-viewer model="DOE_MALI" experiment="ctrl_proj_std" variable="lithk" controls="time" >}}
--
-- Multi-panel:
--   {{< ismip6-viewer panels='[{"model":"DOE_MALI","experiment":"exp05"},{"model":"JPL1_ISSM","experiment":"exp05"}]' variable="lithk" controls="time" >}}
--
-- Options:
--   store_url   - icechunk store URL (required)
--   store_ref   - Store version: branch, tag, or snapshot ID (default: main)
--   model       - Model name (e.g., DOE_MALI)
--   experiment  - Experiment name (e.g., ctrl_proj_std)
--   variable    - Variable to display (e.g., lithk)
--   time        - Initial time index
--   colormap    - Colormap name (viridis, plasma, etc.)
--   vmin        - Color scale minimum
--   vmax        - Color scale maximum
--   panels      - JSON array of panel configs
--   controls    - Controls mode: all, time, none
--   width       - iframe width (default: 100%)
--   height      - iframe height (default: 700)
--   url         - Override viewer base URL

local DEFAULT_URL = "https://models.englacial.org/"

local PARAM_KEYS = {
  "model", "experiment", "variable", "time",
  "colormap", "vmin", "vmax", "panels", "controls",
  "store_url", "store_ref", "group_path", "data_view",
  "grid_width", "grid_height", "cell_size", "x_min", "y_min"
}

local function url_encode(str)
  str = string.gsub(str, "([^%w%-%.%_%~])", function(c)
    return string.format("%%%02X", string.byte(c))
  end)
  return str
end

return {
  ["ismip6-viewer"] = function(args, kwargs, meta)
    if not quarto.doc.is_format("html:js") then
      -- Non-HTML output: render a placeholder
      return pandoc.RawBlock("html",
        '<p><em>[Interactive ISMIP6 viewer — requires HTML output]</em></p>')
    end

    local base_url = pandoc.utils.stringify(kwargs["url"] or DEFAULT_URL)
    local width = pandoc.utils.stringify(kwargs["width"] or "100%")
    local height = pandoc.utils.stringify(kwargs["height"] or "700")

    -- Build query parameters
    local params = { "autoload=true" }

    for _, key in ipairs(PARAM_KEYS) do
      local val = kwargs[key]
      if val then
        val = pandoc.utils.stringify(val)
        if val ~= "" then
          table.insert(params, key .. "=" .. url_encode(val))
        end
      end
    end

    local query = table.concat(params, "&")
    local src = base_url .. "?" .. query

    local html = string.format(
      '<iframe src="%s" width="%s" height="%s" ' ..
      'style="border:1px solid #ccc;border-radius:5px" ' ..
      'frameborder="0" loading="lazy"></iframe>',
      src, width, height
    )

    return pandoc.RawBlock("html", html)
  end
}
